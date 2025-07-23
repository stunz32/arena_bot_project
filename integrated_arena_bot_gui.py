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
from typing import Dict, List, Any, Optional, Tuple, Callable
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
from queue import Queue

# Check GUI availability
GUI_AVAILABLE = True
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    GUI_AVAILABLE = False
    print("WARNING: GUI not available")

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

# Import debug modules
from debug_config import get_debug_config, is_debug_enabled, enable_debug, disable_debug
from visual_debugger import VisualDebugger, create_debug_visualization, save_debug_image
from metrics_logger import MetricsLogger, log_detection_metrics, generate_performance_report
from validation_suite import ValidationSuite, run_full_validation, check_system_health

# Import standard logging
from arena_bot.utils.logging_config import setup_logging, get_logger

# Import CardRefiner for perfect coordinate refinement
from arena_bot.core.card_refiner import CardRefiner

# Import DraftOverlay for visual overlay
from arena_bot.ui.draft_overlay import DraftOverlay

# Import AI v2 Settings System
from arena_bot.ai_v2.settings_manager import get_settings_manager
from arena_bot.ai_v2.settings_dialog import SettingsDialog
from arena_bot.ai_v2.settings_integration import SettingsIntegrator

# Import AI v2 Conversational Coach
from arena_bot.ai_v2.conversational_coach import ConversationalCoach

# Import AI v2 Draft Export/Tracking System
from arena_bot.ai_v2.draft_exporter import get_draft_exporter
from arena_bot.ai_v2.draft_tracking_integration import DraftTrackingIntegrator

# Import AI v2 System Health Monitor
from arena_bot.ai_v2.system_health_monitor import get_health_monitor

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
        
        # Initialize standard logging
        setup_logging()
        self.logger = get_logger(__name__)
        self.logger.info("🚀 Integrated Arena Bot starting up...")
        
        # Step 1: Initialize ALL state variables and components FIRST
        self.running = False
        self.last_full_analysis_result = None
        self.in_draft = False
        self.current_hero = None
        self.draft_picks_count = 0
        self.custom_coordinates = None
        self.last_known_good_coords = None
        self._enable_custom_mode_on_startup = False
        self.analysis_in_progress = False
        self.cache_build_in_progress = False
        self.last_analysis_candidates = [[], [], []]
        self._gui_start_time = time.time()  # For health monitoring
        self.last_detection_result = None
        self.result_queue = Queue()
        self.ui_queue = Queue()

        # Step 2: Initialize all subsystems
        self.init_log_monitoring()
        self.init_ai_advisor()
        self.init_card_detection()
        self.init_settings_system()
        
        self.debug_config = get_debug_config()
        self.visual_debugger = VisualDebugger()
        self.metrics_logger = MetricsLogger()
        self.overlay = None  # Will be created when monitoring starts
        self.ground_truth_data = self.load_ground_truth_data()
        
        self.load_saved_coordinates()
        self.cards_json_loader = self.init_cards_json()
        
        try:
            from arena_bot.data.arena_card_database import ArenaCardDatabase
            self.arena_database = ArenaCardDatabase()
            self.logger.info("✅ Arena database loaded for intelligent card search")
        except Exception as e:
            self.logger.warning(f"⚠️ Arena database not available: {e}")
            self.arena_database = None

        # Step 3: NOW that all attributes exist, build the GUI
        if GUI_AVAILABLE:
            try:
                self.root = tk.Tk()
                self.setup_gui() # This will now succeed
                self.logger.info("✅ GUI setup completed successfully")
            except Exception as e:
                self.logger.error(f"❌ GUI setup failed: {e}")
                self.root = None
        else:
            self.root = None
            self.logger.warning("⚠️ GUI not available, running in headless mode")

        # Step 4: Start background processes AFTER GUI is built
        if self.root:
            self.root.after(100, self._process_ui_queue)
        
        self._start_background_cache_builder()
        
        self.logger.info("🎯 Integrated Arena Bot GUI ready!")
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
    
    def _process_ui_queue(self):
        """Process UI updates from the queue in a thread-safe way."""
        try:
            while not self.ui_queue.empty():
                try:
                    # Get the function and its arguments from the queue
                    func, args, kwargs = self.ui_queue.get_nowait()
                    # Execute the UI update function
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Error processing UI queue: {e}")
        finally:
            # Schedule the next check
            self.root.after(100, self._process_ui_queue)
    
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
        """Initialize the AI draft advisor with AI v2 integration."""
        try:
            # Initialize legacy AI system
            from arena_bot.ai.draft_advisor import get_draft_advisor
            self.advisor = get_draft_advisor()
            print("✅ Legacy AI draft advisor loaded")
        except Exception as e:
            print(f"⚠️ Legacy AI advisor not available: {e}")
            self.advisor = None
        
        # NEW: Initialize AI v2 system
        try:
            from arena_bot.ai_v2.system_integrator import SystemIntegrator
            from arena_bot.ai_v2.hero_selector import HeroSelectionAdvisor
            from arena_bot.ai_v2.grandmaster_advisor import GrandmasterAdvisor
            
            self.ai_v2_integrator = SystemIntegrator()
            self.hero_selector = HeroSelectionAdvisor()
            self.grandmaster_advisor = GrandmasterAdvisor()
            
            print("🎯 AI v2 System loaded:")
            print("   • Hero Selection Advisor with HSReplay integration")
            print("   • Grandmaster Advisor with dimensional scoring")
            print("   • System Integrator with comprehensive error recovery")
            
        except Exception as e:
            print(f"⚠️ AI v2 system not available: {e}")
            self.ai_v2_integrator = None
            self.hero_selector = None
            self.grandmaster_advisor = None
    
    def init_settings_system(self):
        """Initialize the AI v2 settings system and conversational coach."""
        try:
            # Initialize settings manager and integrator
            self.settings_manager = get_settings_manager()
            self.settings_integrator = SettingsIntegrator()
            
            # Initialize conversational coach
            self.conversational_coach = ConversationalCoach()
            
            # Initialize draft export/tracking system
            self.draft_exporter = get_draft_exporter()
            self.draft_tracking_integrator = DraftTrackingIntegrator()
            
            # Initialize system health monitor
            self.health_monitor = get_health_monitor()
            
            # Register main GUI component for health monitoring
            self.health_monitor.register_component(
                'arena_bot_gui',
                health_checker=self._check_gui_health,
                warning_thresholds={'memory_usage_mb': 500, 'response_time_ms': 1000},
                critical_thresholds={'memory_usage_mb': 1000, 'response_time_ms': 5000}
            )
            
            # Register AI v2 components with settings integrator
            if hasattr(self, 'grandmaster_advisor') and self.grandmaster_advisor:
                self.settings_integrator.register_component('grandmaster_advisor', self.grandmaster_advisor)
                
                # Inject advisor reference into the coach for contextual advice
                self.conversational_coach.grandmaster_advisor = self.grandmaster_advisor
                
            if hasattr(self, 'hero_selector') and self.hero_selector:
                self.settings_integrator.register_component('hero_selector', self.hero_selector)
                
                # Inject hero selector reference into the coach
                self.conversational_coach.hero_selector = self.hero_selector
            
            if self.conversational_coach:
                self.settings_integrator.register_component('conversational_coach', self.conversational_coach)
            
            print("✅ Settings system, conversational coach, draft export system, and health monitor initialized")
            self.settings_available = True
            self.coach_available = True
            self.export_available = True
            self.health_available = True
            
        except Exception as e:
            print(f"⚠️ Settings system not available: {e}")
            self.settings_available = False
            self.coach_available = False
            self.export_available = False
            self.health_available = False
            self.settings_manager = None
            self.settings_integrator = None
            self.conversational_coach = None
            self.draft_exporter = None
            self.draft_tracking_integrator = None
            self.health_monitor = None
    
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
        """Setup log monitoring callbacks to be thread-safe."""
        if not self.log_monitor:
            return

        # Initialize draft phase tracking
        self.current_draft_phase = "waiting"  # waiting, hero_selection, card_picks, complete
        self.selected_hero_class = None
        self.hero_recommendation = None

        def on_draft_start():
            # Use the queue to schedule UI updates on the main thread
            self.ui_queue.put((self.log_text, (f"\n{'🎯' * 50}\n🎯 ARENA DRAFT STARTED!",), {}))
            self.in_draft = True
            self.draft_picks_count = 0
            self.ui_queue.put((self.update_status, ("Arena Draft Active",), {}))

        def on_draft_complete(picks):
            self.ui_queue.put((self.log_text, (f"\n{'🏆' * 50}\n🏆 ARENA DRAFT COMPLETED!\n🏆 Total picks: {len(picks)}\n{'🏆' * 50}",), {}))
            self.in_draft = False
            self.ui_queue.put((self.update_status, ("Draft Complete",), {}))
            self.ui_queue.put((self._show_draft_summary, (), {}))

        def on_game_state_change(old_state, new_state):
            self.ui_queue.put((self.log_text, (f"\n🎮 GAME STATE: {old_state.value} → {new_state.value}",), {}))
            self.in_draft = (new_state.value == "Arena Draft")
            self.ui_queue.put((self.update_status, (f"Game State: {new_state.value}",), {}))

        def on_draft_pick(pick):
            self.draft_picks_count += 1
            card_name = self.get_card_name(pick.card_code)
            self.ui_queue.put((self.log_text, (f"\n📋 PICK #{self.draft_picks_count}: {card_name}",), {}))
            if pick.is_premium:
                self.ui_queue.put((self.log_text, ("   ✨ GOLDEN CARD!",), {}))
            
            # --- FIX: Trigger automatic screenshot analysis ---
            self.ui_queue.put((self.log_text, ("🤖 Log event triggered automatic screenshot analysis...",), {}))
            self.ui_queue.put((self.manual_screenshot, (), {}))

        def on_hero_choices_ready(hero_data):
            self.ui_queue.put((self.log_text, (f"\n{'👑' * 50}\n👑 HERO CHOICES READY!",), {}))
            hero_classes = hero_data.get('hero_classes', [])
            if hero_classes:
                self.ui_queue.put((self.log_text, (f"👑 Available heroes: {', '.join(hero_classes)}",), {}))
                # Schedule the AI analysis to be called from the main thread
                self.ui_queue.put((self._handle_enhanced_hero_selection, (hero_classes, hero_data), {}))

        def on_card_choices_ready(choice_data):
            self.ui_queue.put((self.log_text, (f"\n🤖 CARD CHOICES READY - Triggering automatic analysis...",), {}))
            # Trigger automatic screenshot analysis when new card choices appear
            self.ui_queue.put((self.manual_screenshot, (), {}))
        
        # Set up all callbacks
        self.log_monitor.on_draft_start = on_draft_start
        self.log_monitor.on_draft_complete = on_draft_complete
        self.log_monitor.on_game_state_change = on_game_state_change
        self.log_monitor.on_draft_pick = on_draft_pick
        self.log_monitor.on_hero_choices_ready = on_hero_choices_ready
        self.log_monitor.on_card_choices_ready = on_card_choices_ready
    
    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title("🎯 Integrated Arena Bot - Complete GUI")
        self.root.geometry("1800x1200")  # Even larger size for proper card visibility
        self.root.configure(bg='#2C3E50')
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
        # Create menu bar
        self.create_menu_bar()
        
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
        self.detection_method = tk.StringVar(value="hybrid_cascade")
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
        
        # Detection method status label - makes current selection obvious
        self.detection_status_label = tk.Label(
            control_frame,
            text="🔄 HYBRID CASCADE",
            bg='#27AE60',  # Green background to show it's active
            fg='white',
            font=('Arial', 8, 'bold'),
            relief='raised',
            bd=2
        )
        self.detection_status_label.pack(side='left', padx=2)
        
        # Debug controls
        self.debug_enabled = tk.BooleanVar(value=True)
        self.debug_enabled.trace_add("write", self.toggle_debug_mode)
        debug_check = tk.Checkbutton(
            control_frame,
            text="🐛 DEBUG",
            variable=self.debug_enabled,
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
        
        # Validation suite button
        self.validation_btn = tk.Button(
            control_frame,
            text="🧪 VALIDATE",
            command=self.run_validation_suite,
            bg='#27AE60',
            fg='white',
            font=('Arial', 9),
            relief='raised',
            bd=2
        )
        self.validation_btn.pack(side='left', padx=5)
        
        # NEW: Unified Statistics button
        self.stats_btn = tk.Button(
            control_frame,
            text="📈 AI v2 STATS",
            command=self.show_unified_statistics,
            bg='#E74C3C',
            fg='white',
            font=('Arial', 9),
            relief='raised',
            bd=2
        )
        self.stats_btn.pack(side='left', padx=5)
        
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

        # Advanced manual correction button
        self.advanced_correction_btn = tk.Button(
            control_frame,
            text="🔧 MANUAL CORRECTIONS",
            command=self.open_advanced_correction_center,
            bg='#E67E22',
            fg='white',
            font=('Arial', 9),
            relief='raised',
            bd=2
        )
        self.advanced_correction_btn.pack(side='left', padx=5)
        
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
        self.use_ultimate_detection = tk.BooleanVar(value=True)
        self.use_ultimate_detection.trace_add("write", self.toggle_ultimate_detection)
        self.ultimate_detection_btn = tk.Checkbutton(
            control_frame,
            text="🚀 Ultimate Detection",
            variable=self.use_ultimate_detection,
            bg='#2C3E50',
            fg='#E74C3C',  # Red color to indicate advanced feature
            selectcolor='#34495E',
            font=('Arial', 8, 'bold'),
            relief='flat'
        )
        # Always show Ultimate Detection checkbox - disable if not available
        if self.ultimate_detector:
            self.ultimate_detection_btn.pack(side='left', padx=5)
        else:
            self.ultimate_detection_btn.config(state='disabled')
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
        self.use_phash_detection = tk.BooleanVar(value=True)  # Default enabled for speed
        self.use_phash_detection.trace_add("write", self.toggle_phash_detection)
        self.phash_detection_btn = tk.Checkbutton(
            control_frame,
            text="⚡ pHash Detection",
            variable=self.use_phash_detection,
            bg='#2C3E50',
            fg='#E67E22',  # Electric orange color to indicate speed
            selectcolor='#34495E',
            font=('Arial', 8, 'bold'),
            relief='flat'
        )
        # Always show pHash Detection checkbox - disable if not available  
        if self.phash_matcher:
            self.phash_detection_btn.pack(side='left', padx=5)
        else:
            self.phash_detection_btn.config(state='disabled')
            self.phash_detection_btn.pack(side='left', padx=5)
        
        # Enable custom mode if coordinates were loaded during startup
        if self._enable_custom_mode_on_startup:
            self.use_custom_coords.set(True)
            print("🎯 Custom coordinate mode enabled (coordinates loaded from previous session)")
        
        # Update coordinate status display
        self.update_coordinate_status()
        
        # NEW: Draft Progression Display
        self.setup_draft_progression_display()
        
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
        
        # AI Coach Chat Interface
        self.setup_ai_coach_interface()
        
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
        
        # Setup draft review panel
        self.setup_draft_review_panel()
    
    def create_menu_bar(self):
        """Create the menu bar with Settings option."""
        try:
            # Create menu bar
            self.menubar = tk.Menu(self.root)
            self.root.config(menu=self.menubar)
            
            # Settings menu
            settings_menu = tk.Menu(self.menubar, tearoff=0)
            self.menubar.add_cascade(label="Settings", menu=settings_menu)
            
            # Settings dialog option
            if hasattr(self, 'settings_available') and self.settings_available:
                settings_menu.add_command(
                    label="⚙️ AI v2 Settings...",
                    command=self.open_settings_dialog
                )
            else:
                settings_menu.add_command(
                    label="⚙️ AI v2 Settings... (Unavailable)",
                    command=lambda: messagebox.showwarning(
                        "Settings Unavailable", 
                        "Settings system not available. Please check that AI v2 components are loaded."
                    ),
                    state='disabled'
                )
            
            # Export settings
            settings_menu.add_separator()
            settings_menu.add_command(
                label="📤 Export Settings...",
                command=self.export_settings
            )
            settings_menu.add_command(
                label="📥 Import Settings...",
                command=self.import_settings
            )
            
            # Reset settings
            settings_menu.add_separator()
            settings_menu.add_command(
                label="🔄 Reset to Defaults",
                command=self.reset_settings
            )
            
            # Draft menu
            if hasattr(self, 'export_available') and self.export_available:
                draft_menu = tk.Menu(self.menubar, tearoff=0)
                self.menubar.add_cascade(label="Draft", menu=draft_menu)
                
                # Draft tracking options
                draft_menu.add_command(
                    label="▶️ Start Draft Tracking",
                    command=self.start_draft_tracking
                )
                draft_menu.add_command(
                    label="⏹️ Stop Draft Tracking",
                    command=self.stop_draft_tracking
                )
                draft_menu.add_command(
                    label="📊 Draft Status",
                    command=self.show_draft_status
                )
                
                draft_menu.add_separator()
                
                # Export options
                draft_menu.add_command(
                    label="💾 Export Current Draft...",
                    command=self.export_current_draft
                )
                draft_menu.add_command(
                    label="📋 View Export Dialog...",
                    command=self.show_export_dialog
                )
                
                draft_menu.add_separator()
                
                # Configuration options
                draft_menu.add_command(
                    label="⚙️ Configure Auto-Tracking...",
                    command=self.configure_draft_tracking
                )
            
            # System Health menu (add after Draft menu)
            if hasattr(self, 'health_available') and self.health_available:
                health_menu = tk.Menu(self.menubar, tearoff=0)
                self.menubar.add_cascade(label="System Health", menu=health_menu)
                
                # Health monitoring options
                health_menu.add_command(
                    label="🏥 System Health Status",
                    command=self.show_system_health
                )
                health_menu.add_command(
                    label="📊 Export Health Report",
                    command=self.export_health_report
                )
                
                health_menu.add_separator()
                
                # Health monitoring tools
                health_menu.add_command(
                    label="🔄 Run Health Check",
                    command=self.run_manual_health_check
                )
                health_menu.add_command(
                    label="⚙️ Configure Monitoring",
                    command=self.configure_health_monitoring
                )
            
        except Exception as e:
            print(f"⚠️ Failed to create menu bar: {e}")
    
    def open_settings_dialog(self):
        """Open the settings configuration dialog."""
        try:
            if not hasattr(self, 'settings_available') or not self.settings_available:
                messagebox.showerror(
                    "Settings Unavailable",
                    "Settings system is not available. Please check the console for errors."
                )
                return
            
            # Create and show settings dialog
            dialog = SettingsDialog(self.root, on_settings_changed=self.on_settings_changed)
            
        except Exception as e:
            self.logger.error(f"Failed to open settings dialog: {e}")
            messagebox.showerror("Error", f"Failed to open settings dialog: {e}")
    
    def on_settings_changed(self):
        """Called when settings are changed to apply them to AI components."""
        try:
            if hasattr(self, 'settings_integrator') and self.settings_integrator:
                # The SettingsIntegrator will automatically apply settings to registered components
                self.log_text("✅ Settings updated and applied to AI v2 components")
            else:
                self.log_text("⚠️ Settings updated but integrator not available")
                
        except Exception as e:
            self.logger.error(f"Error applying settings changes: {e}")
            self.log_text(f"❌ Error applying settings: {e}")
    
    def export_settings(self):
        """Export current settings to a file."""
        try:
            if not hasattr(self, 'settings_manager') or not self.settings_manager:
                messagebox.showerror("Error", "Settings system not available")
                return
            
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Settings"
            )
            
            if filename:
                if self.settings_manager.export_settings(filename):
                    messagebox.showinfo("Success", f"Settings exported to {filename}")
                else:
                    messagebox.showerror("Error", "Failed to export settings")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export settings: {e}")
    
    def import_settings(self):
        """Import settings from a file."""
        try:
            if not hasattr(self, 'settings_manager') or not self.settings_manager:
                messagebox.showerror("Error", "Settings system not available")
                return
            
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Import Settings"
            )
            
            if filename:
                if self.settings_manager.import_settings(filename):
                    messagebox.showinfo("Success", "Settings imported successfully")
                    self.on_settings_changed()  # Apply the imported settings
                else:
                    messagebox.showerror("Error", "Failed to import settings")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import settings: {e}")
    
    def reset_settings(self):
        """Reset all settings to defaults."""
        try:
            if not hasattr(self, 'settings_manager') or not self.settings_manager:
                messagebox.showerror("Error", "Settings system not available")
                return
            
            result = messagebox.askyesno(
                "Reset Settings",
                "Are you sure you want to reset all settings to defaults? This cannot be undone."
            )
            
            if result:
                self.settings_manager.reset_to_defaults()
                if self.settings_manager.save_settings():
                    messagebox.showinfo("Success", "Settings reset to defaults")
                    self.on_settings_changed()  # Apply the reset settings
                else:
                    messagebox.showerror("Error", "Failed to save reset settings")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset settings: {e}")
    
    # =========================================================================
    # Draft Export/Tracking Methods
    # =========================================================================
    
    def start_draft_tracking(self):
        """Start tracking a new draft session."""
        try:
            if not hasattr(self, 'export_available') or not self.export_available:
                messagebox.showerror(
                    "Draft Export Unavailable",
                    "Draft tracking system is not available. Please check the console for errors."
                )
                return
            
            if not hasattr(self, 'draft_tracking_integrator') or not self.draft_tracking_integrator:
                messagebox.showerror("Error", "Draft tracking system not initialized")
                return
            
            # Start tracking
            draft_id = self.draft_tracking_integrator.start_draft_tracking()
            
            messagebox.showinfo(
                "Draft Tracking Started",
                f"Draft tracking has been started.\nDraft ID: {draft_id}\n\nThe system will automatically track your hero selection and card picks."
            )
            
            self.log_text(f"✅ Started draft tracking: {draft_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start draft tracking: {e}")
            messagebox.showerror("Error", f"Failed to start draft tracking: {e}")
    
    def stop_draft_tracking(self):
        """Stop the current draft tracking session."""
        try:
            if not hasattr(self, 'export_available') or not self.export_available:
                messagebox.showerror("Error", "Draft tracking system not available")
                return
            
            if not hasattr(self, 'draft_tracking_integrator') or not self.draft_tracking_integrator:
                messagebox.showerror("Error", "Draft tracking system not initialized")
                return
            
            # Get current status first
            status = self.draft_tracking_integrator.get_current_draft_status()
            
            if not status['draft_active']:
                messagebox.showinfo("No Active Draft", "No draft is currently being tracked.")
                return
            
            # Confirm stopping
            result = messagebox.askyesno(
                "Stop Draft Tracking",
                f"Are you sure you want to stop tracking the current draft?\n\nDraft ID: {status['draft_id']}\nCards picked: {status['card_picks_count']}/30\n\nThis will complete the draft and make it available for export."
            )
            
            if result:
                draft_summary = self.draft_tracking_integrator.complete_draft_tracking()
                
                if draft_summary:
                    messagebox.showinfo(
                        "Draft Tracking Stopped",
                        f"Draft tracking completed successfully!\n\nDraft ID: {draft_summary.draft_id}\nTotal picks: {len(draft_summary.pick_history)}\n\nYou can now export this draft from the Draft menu."
                    )
                    self.log_text(f"✅ Completed draft tracking: {draft_summary.draft_id}")
                else:
                    messagebox.showerror("Error", "Failed to complete draft tracking")
            
        except Exception as e:
            self.logger.error(f"Failed to stop draft tracking: {e}")
            messagebox.showerror("Error", f"Failed to stop draft tracking: {e}")
    
    def show_draft_status(self):
        """Show current draft tracking status."""
        try:
            if not hasattr(self, 'export_available') or not self.export_available:
                messagebox.showerror("Error", "Draft tracking system not available")
                return
            
            if not hasattr(self, 'draft_tracking_integrator') or not self.draft_tracking_integrator:
                messagebox.showerror("Error", "Draft tracking system not initialized")
                return
            
            # Get current status
            status = self.draft_tracking_integrator.get_current_draft_status()
            statistics = self.draft_tracking_integrator.get_draft_statistics()
            
            # Format status message
            if status['draft_active']:
                status_msg = f"""🟢 ACTIVE DRAFT TRACKING
                
Draft ID: {status['draft_id']}
Started: {status['session_start_time']}
Hero Selected: {'✅' if status['hero_selection_completed'] else '❌'}
Cards Picked: {status['card_picks_count']}/30
Auto-Tracking: {'Enabled' if status['auto_tracking_enabled'] else 'Disabled'}
Auto-Export: {'Enabled' if status['auto_export_enabled'] else 'Disabled'}

SYSTEM STATISTICS
Total Drafts Tracked: {statistics.get('total_drafts', 0)}
Total Exports: {statistics.get('total_exports', 0)}
Available Formats: {', '.join(status['export_formats'])}"""
            else:
                status_msg = f"""⚪ NO ACTIVE DRAFT
                
No draft is currently being tracked.

SYSTEM STATISTICS
Total Drafts Tracked: {statistics.get('total_drafts', 0)}
Total Exports: {statistics.get('total_exports', 0)}
Auto-Tracking: {'Enabled' if status['auto_tracking_enabled'] else 'Disabled'}
Auto-Export: {'Enabled' if status['auto_export_enabled'] else 'Disabled'}
Available Formats: {', '.join(status['export_formats'])}

Start a new draft from the Draft menu to begin tracking."""
            
            messagebox.showinfo("Draft Tracking Status", status_msg)
            
        except Exception as e:
            self.logger.error(f"Failed to show draft status: {e}")
            messagebox.showerror("Error", f"Failed to show draft status: {e}")
    
    def export_current_draft(self):
        """Export the current or most recent draft."""
        try:
            if not hasattr(self, 'export_available') or not self.export_available:
                messagebox.showerror("Error", "Draft export system not available")
                return
            
            if not hasattr(self, 'draft_tracking_integrator') or not self.draft_tracking_integrator:
                messagebox.showerror("Error", "Draft tracking system not initialized")
                return
            
            # Get current status
            status = self.draft_tracking_integrator.get_current_draft_status()
            
            if status['draft_active']:
                # Current draft is active - ask if they want to complete it first
                result = messagebox.askyesno(
                    "Complete Draft First?",
                    f"A draft is currently active (ID: {status['draft_id']}).\n\nWould you like to complete it first before exporting?\n\nClick 'Yes' to complete and export, or 'No' to export the current state as a preview."
                )
                
                if result:
                    # Complete the draft first
                    draft_summary = self.draft_tracking_integrator.complete_draft_tracking()
                    if not draft_summary:
                        messagebox.showerror("Error", "Failed to complete draft")
                        return
                else:
                    # Export current state as preview
                    messagebox.showinfo(
                        "Preview Export",
                        "The export dialog will show a preview of the current draft state.\n\nNote: This is not a complete draft export."
                    )
            
            # Show export dialog
            self.show_export_dialog()
            
        except Exception as e:
            self.logger.error(f"Failed to export current draft: {e}")
            messagebox.showerror("Error", f"Failed to export current draft: {e}")
    
    def show_export_dialog(self):
        """Show the draft export dialog."""
        try:
            if not hasattr(self, 'export_available') or not self.export_available:
                messagebox.showerror("Error", "Draft export system not available")
                return
            
            if not hasattr(self, 'draft_tracking_integrator') or not self.draft_tracking_integrator:
                messagebox.showerror("Error", "Draft tracking system not initialized")
                return
            
            # Show export dialog
            self.draft_tracking_integrator.show_export_dialog(parent=self.root)
            
        except Exception as e:
            self.logger.error(f"Failed to show export dialog: {e}")
            messagebox.showerror("Error", f"Failed to show export dialog: {e}")
    
    def configure_draft_tracking(self):
        """Configure automatic draft tracking settings."""
        try:
            if not hasattr(self, 'export_available') or not self.export_available:
                messagebox.showerror("Error", "Draft tracking system not available")
                return
            
            if not hasattr(self, 'draft_tracking_integrator') or not self.draft_tracking_integrator:
                messagebox.showerror("Error", "Draft tracking system not initialized")
                return
            
            # Create configuration dialog
            config_dialog = tk.Toplevel(self.root)
            config_dialog.title("Draft Tracking Configuration")
            config_dialog.geometry("400x300")
            config_dialog.configure(bg='#2C3E50')
            config_dialog.transient(self.root)
            config_dialog.grab_set()
            
            # Get current status
            status = self.draft_tracking_integrator.get_current_draft_status()
            
            # Configuration options
            main_frame = tk.Frame(config_dialog, bg='#2C3E50')
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            tk.Label(
                main_frame,
                text="⚙️ Draft Tracking Configuration",
                font=('Arial', 14, 'bold'),
                fg='#ECF0F1',
                bg='#2C3E50'
            ).pack(pady=(0, 20))
            
            # Auto-tracking option
            auto_tracking_var = tk.BooleanVar(value=status['auto_tracking_enabled'])
            tk.Checkbutton(
                main_frame,
                text="Enable automatic draft tracking",
                variable=auto_tracking_var,
                fg='#ECF0F1',
                bg='#2C3E50',
                selectcolor='#34495E',
                font=('Arial', 10)
            ).pack(anchor='w', pady=5)
            
            # Auto-export option
            auto_export_var = tk.BooleanVar(value=status['auto_export_enabled'])
            tk.Checkbutton(
                main_frame,
                text="Enable automatic export on draft completion",
                variable=auto_export_var,
                fg='#ECF0F1',
                bg='#2C3E50',
                selectcolor='#34495E',
                font=('Arial', 10)
            ).pack(anchor='w', pady=5)
            
            # Export formats
            tk.Label(
                main_frame,
                text="Export Formats:",
                fg='#ECF0F1',
                bg='#2C3E50',
                font=('Arial', 10, 'bold')
            ).pack(anchor='w', pady=(15, 5))
            
            format_vars = {}
            available_formats = ['json', 'csv', 'html', 'txt']
            current_formats = status['export_formats']
            
            for fmt in available_formats:
                var = tk.BooleanVar(value=fmt in current_formats)
                tk.Checkbutton(
                    main_frame,
                    text=fmt.upper(),
                    variable=var,
                    fg='#ECF0F1',
                    bg='#2C3E50',
                    selectcolor='#34495E',
                    font=('Arial', 9)
                ).pack(anchor='w', padx=20, pady=2)
                format_vars[fmt] = var
            
            # Buttons
            button_frame = tk.Frame(main_frame, bg='#2C3E50')
            button_frame.pack(fill='x', pady=(20, 0))
            
            def save_config():
                # Get selected formats
                selected_formats = [fmt for fmt, var in format_vars.items() if var.get()]
                if not selected_formats:
                    messagebox.showwarning("No Formats", "Please select at least one export format.")
                    return
                
                # Apply configuration
                self.draft_tracking_integrator.configure_auto_tracking(
                    enabled=auto_tracking_var.get(),
                    auto_export=auto_export_var.get(),
                    export_formats=selected_formats
                )
                
                messagebox.showinfo("Success", "Draft tracking configuration saved!")
                config_dialog.destroy()
            
            tk.Button(
                button_frame,
                text="💾 Save",
                command=save_config,
                bg='#27AE60',
                fg='white',
                font=('Arial', 10)
            ).pack(side='right', padx=5)
            
            tk.Button(
                button_frame,
                text="❌ Cancel",
                command=config_dialog.destroy,
                bg='#E74C3C',
                fg='white',
                font=('Arial', 10)
            ).pack(side='right', padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to configure draft tracking: {e}")
            messagebox.showerror("Error", f"Failed to configure draft tracking: {e}")
    
    # =========================================================================
    # System Health Monitor Methods
    # =========================================================================
    
    def _check_gui_health(self) -> Dict[str, Any]:
        """Health check for the main GUI component."""
        try:
            import psutil
            import os
            
            # Get current process info
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Check if GUI is responsive
            gui_responsive = True
            try:
                if hasattr(self, 'root') and self.root:
                    self.root.update_idletasks()
            except:
                gui_responsive = False
            
            # Calculate uptime
            uptime_seconds = time.time() - getattr(self, '_gui_start_time', time.time())
            
            # Determine health status
            from arena_bot.ai_v2.system_health_monitor import HealthStatus
            
            status = HealthStatus.HEALTHY
            if memory_mb > 1000 or not gui_responsive:
                status = HealthStatus.CRITICAL
            elif memory_mb > 500:
                status = HealthStatus.WARNING
            
            return {
                'status': status.value,
                'metrics': {
                    'memory_usage_mb': memory_mb,
                    'gui_responsive': gui_responsive,
                    'uptime_seconds': uptime_seconds,
                    'threads_active': threading.active_count()
                }
            }
            
        except Exception as e:
            from arena_bot.ai_v2.system_health_monitor import HealthStatus
            return {
                'status': HealthStatus.CRITICAL.value,
                'error': f"GUI health check failed: {e}"
            }
    
    def show_system_health(self):
        """Show comprehensive system health status."""
        try:
            if not hasattr(self, 'health_available') or not self.health_available:
                messagebox.showerror("Error", "System health monitoring not available")
                return
            
            if not hasattr(self, 'health_monitor') or not self.health_monitor:
                messagebox.showerror("Error", "Health monitor not initialized")
                return
            
            # Get system health data
            health_data = self.health_monitor.get_system_health()
            
            # Create health status window
            health_window = tk.Toplevel(self.root)
            health_window.title("System Health Monitor")
            health_window.geometry("800x600")
            health_window.configure(bg='#2C3E50')
            health_window.transient(self.root)
            
            # Main frame with scrollbar
            main_frame = tk.Frame(health_window, bg='#2C3E50')
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            canvas = tk.Canvas(main_frame, bg='#2C3E50')
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg='#2C3E50')
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Title
            tk.Label(
                scrollable_frame,
                text="🏥 System Health Monitor",
                font=('Arial', 16, 'bold'),
                fg='#ECF0F1',
                bg='#2C3E50'
            ).pack(pady=(0, 20))
            
            # Overall status
            status_color = {
                'healthy': '#27AE60',
                'warning': '#F39C12', 
                'critical': '#E74C3C',
                'unknown': '#95A5A6'
            }.get(health_data['overall_status'], '#95A5A6')
            
            status_frame = tk.Frame(scrollable_frame, bg='#34495E', relief='ridge', bd=2)
            status_frame.pack(fill='x', pady=10)
            
            tk.Label(
                status_frame,
                text=f"Overall Status: {health_data['overall_status'].upper()}",
                font=('Arial', 14, 'bold'),
                fg=status_color,
                bg='#34495E'
            ).pack(pady=10)
            
            # System metrics
            metrics_frame = tk.LabelFrame(
                scrollable_frame,
                text="System Metrics",
                font=('Arial', 12, 'bold'),
                fg='#ECF0F1',
                bg='#34495E'
            )
            metrics_frame.pack(fill='x', pady=10)
            
            metrics_text = f"""System Uptime: {health_data['system_uptime_hours']:.1f} hours
Total Components: {health_data['total_components']}
Healthy Components: {health_data['healthy_components']}
Warning Components: {health_data['warning_components']}
Critical Components: {health_data['critical_components']}
Active Alerts: {health_data['system_metrics'].get('active_alerts', 0)}
Total Alerts: {health_data['system_metrics'].get('total_alerts', 0)}"""
            
            tk.Label(
                metrics_frame,
                text=metrics_text,
                font=('Arial', 10),
                fg='#ECF0F1',
                bg='#34495E',
                justify='left'
            ).pack(padx=20, pady=10, anchor='w')
            
            # Component health
            components_frame = tk.LabelFrame(
                scrollable_frame,
                text="Component Health",
                font=('Arial', 12, 'bold'),
                fg='#ECF0F1',
                bg='#34495E'
            )
            components_frame.pack(fill='x', pady=10)
            
            for comp_name, comp_data in health_data['component_health'].items():
                comp_color = {
                    'healthy': '#27AE60',
                    'warning': '#F39C12',
                    'critical': '#E74C3C',
                    'unknown': '#95A5A6',
                    'offline': '#7F8C8D'
                }.get(comp_data['status'], '#95A5A6')
                
                comp_text = f"{comp_name}: {comp_data['status'].upper()} (Errors: {comp_data['error_count']}, Uptime: {comp_data['uptime_hours']:.1f}h)"
                
                tk.Label(
                    components_frame,
                    text=comp_text,
                    font=('Arial', 9),
                    fg=comp_color,
                    bg='#34495E'
                ).pack(anchor='w', padx=20, pady=2)
            
            # Recent alerts
            if health_data['recent_alerts']:
                alerts_frame = tk.LabelFrame(
                    scrollable_frame,
                    text="Recent Alerts",
                    font=('Arial', 12, 'bold'),
                    fg='#ECF0F1',
                    bg='#34495E'
                )
                alerts_frame.pack(fill='x', pady=10)
                
                for alert in health_data['recent_alerts'][-5:]:  # Last 5 alerts
                    alert_color = {
                        'info': '#3498DB',
                        'warning': '#F39C12',
                        'error': '#E67E22',
                        'critical': '#E74C3C'
                    }.get(alert['severity'], '#95A5A6')
                    
                    alert_text = f"[{alert['severity'].upper()}] {alert['component']}: {alert['message']}"
                    
                    tk.Label(
                        alerts_frame,
                        text=alert_text,
                        font=('Arial', 8),
                        fg=alert_color,
                        bg='#34495E',
                        wraplength=700
                    ).pack(anchor='w', padx=20, pady=2)
            
            # Buttons
            button_frame = tk.Frame(scrollable_frame, bg='#2C3E50')
            button_frame.pack(fill='x', pady=20)
            
            tk.Button(
                button_frame,
                text="🔄 Refresh",
                command=lambda: (health_window.destroy(), self.show_system_health()),
                bg='#3498DB',
                fg='white',
                font=('Arial', 10)
            ).pack(side='left', padx=5)
            
            tk.Button(
                button_frame,
                text="📊 Export Report",
                command=lambda: self.export_health_report(),
                bg='#2ECC71',
                fg='white',
                font=('Arial', 10)
            ).pack(side='left', padx=5)
            
            tk.Button(
                button_frame,
                text="❌ Close",
                command=health_window.destroy,
                bg='#E74C3C',
                fg='white',
                font=('Arial', 10)
            ).pack(side='right', padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to show system health: {e}")
            messagebox.showerror("Error", f"Failed to show system health: {e}")
    
    def export_health_report(self):
        """Export system health report to file."""
        try:
            if not hasattr(self, 'health_available') or not self.health_available:
                messagebox.showerror("Error", "System health monitoring not available")
                return
            
            # Get health report
            report = self.health_monitor.export_health_report('json')
            
            # Save to file
            from tkinter import filedialog
            import json
            
            filename = filedialog.asksaveasfilename(
                title="Export Health Report",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                
                messagebox.showinfo(
                    "Export Complete",
                    f"Health report exported successfully to:\n{filename}"
                )
                self.log_text(f"✅ Health report exported: {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to export health report: {e}")
            messagebox.showerror("Error", f"Failed to export health report: {e}")
    
    def run_manual_health_check(self):
        """Run manual health check on all components."""
        try:
            if not hasattr(self, 'health_available') or not self.health_available:
                messagebox.showerror("Error", "System health monitoring not available")
                return
            
            # Show progress dialog
            progress_dialog = tk.Toplevel(self.root)
            progress_dialog.title("Running Health Check")
            progress_dialog.geometry("400x200")
            progress_dialog.configure(bg='#2C3E50')
            progress_dialog.transient(self.root)
            progress_dialog.grab_set()
            
            tk.Label(
                progress_dialog,
                text="🔄 Running Health Check...",
                font=('Arial', 12, 'bold'),
                fg='#ECF0F1',
                bg='#2C3E50'
            ).pack(pady=20)
            
            progress_text = scrolledtext.ScrolledText(
                progress_dialog,
                height=8,
                bg='#34495E',
                fg='#ECF0F1',
                font=('Consolas', 9)
            )
            progress_text.pack(fill='both', expand=True, padx=20, pady=10)
            
            # Run health checks
            def run_checks():
                try:
                    progress_text.insert(tk.END, "Starting health check...\n")
                    progress_text.update()
                    
                    # Force health check on all components
                    results = self.health_monitor.run_health_check(force=True)
                    
                    for component, result in results.items():
                        status = result.get('status', 'unknown')
                        response_time = result.get('response_time_ms', 0)
                        error = result.get('error')
                        
                        if error:
                            progress_text.insert(tk.END, f"❌ {component}: {status} - {error}\n")
                        else:
                            progress_text.insert(tk.END, f"✅ {component}: {status} ({response_time:.1f}ms)\n")
                        
                        progress_text.update()
                        progress_text.see(tk.END)
                    
                    progress_text.insert(tk.END, "\nHealth check completed!\n")
                    progress_text.update()
                    
                    # Add close button
                    tk.Button(
                        progress_dialog,
                        text="✅ Close",
                        command=progress_dialog.destroy,
                        bg='#27AE60',
                        fg='white',
                        font=('Arial', 10)
                    ).pack(pady=10)
                    
                except Exception as e:
                    progress_text.insert(tk.END, f"❌ Health check failed: {e}\n")
                    progress_text.update()
            
            # Run checks in background
            threading.Thread(target=run_checks, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Failed to run manual health check: {e}")
            messagebox.showerror("Error", f"Failed to run manual health check: {e}")
    
    def configure_health_monitoring(self):
        """Configure health monitoring settings."""
        try:
            if not hasattr(self, 'health_available') or not self.health_available:
                messagebox.showerror("Error", "System health monitoring not available")
                return
            
            # Create configuration dialog
            config_dialog = tk.Toplevel(self.root)
            config_dialog.title("Health Monitoring Configuration")
            config_dialog.geometry("500x400")
            config_dialog.configure(bg='#2C3E50')
            config_dialog.transient(self.root)
            config_dialog.grab_set()
            
            main_frame = tk.Frame(config_dialog, bg='#2C3E50')
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            tk.Label(
                main_frame,
                text="⚙️ Health Monitoring Configuration",
                font=('Arial', 14, 'bold'),
                fg='#ECF0F1',
                bg='#2C3E50'
            ).pack(pady=(0, 20))
            
            # Current system status
            system_health = self.health_monitor.get_system_health()
            
            status_frame = tk.LabelFrame(
                main_frame,
                text="Current Status",
                font=('Arial', 12, 'bold'),
                fg='#ECF0F1',
                bg='#34495E'
            )
            status_frame.pack(fill='x', pady=10)
            
            status_text = f"""Overall Status: {system_health['overall_status'].upper()}
Monitoring Enabled: {'Yes' if system_health['monitoring_enabled'] else 'No'}
Components Monitored: {system_health['total_components']}
Active Alerts: {system_health['system_metrics'].get('active_alerts', 0)}
System Uptime: {system_health['system_uptime_hours']:.1f} hours"""
            
            tk.Label(
                status_frame,
                text=status_text,
                font=('Arial', 10),
                fg='#ECF0F1',
                bg='#34495E',
                justify='left'
            ).pack(padx=20, pady=10, anchor='w')
            
            # Configuration options
            options_frame = tk.LabelFrame(
                main_frame,
                text="Configuration Options",
                font=('Arial', 12, 'bold'),
                fg='#ECF0F1',
                bg='#34495E'
            )
            options_frame.pack(fill='x', pady=10)
            
            # Monitoring interval
            tk.Label(
                options_frame,
                text="Monitoring Interval (seconds):",
                fg='#ECF0F1',
                bg='#34495E',
                font=('Arial', 10)
            ).grid(row=0, column=0, sticky='w', padx=20, pady=5)
            
            interval_var = tk.IntVar(value=self.health_monitor.monitoring_interval)
            interval_scale = tk.Scale(
                options_frame,
                from_=10,
                to=300,
                orient='horizontal',
                variable=interval_var,
                fg='#ECF0F1',
                bg='#34495E'
            )
            interval_scale.grid(row=0, column=1, sticky='ew', padx=20, pady=5)
            
            # Enable/disable monitoring
            monitoring_var = tk.BooleanVar(value=self.health_monitor.monitoring_enabled)
            tk.Checkbutton(
                options_frame,
                text="Enable continuous monitoring",
                variable=monitoring_var,
                fg='#ECF0F1',
                bg='#34495E',
                selectcolor='#2C3E50',
                font=('Arial', 10)
            ).grid(row=1, column=0, columnspan=2, sticky='w', padx=20, pady=5)
            
            options_frame.columnconfigure(1, weight=1)
            
            # Buttons
            button_frame = tk.Frame(main_frame, bg='#2C3E50')
            button_frame.pack(fill='x', pady=20)
            
            def save_config():
                try:
                    # Update monitoring interval
                    self.health_monitor.monitoring_interval = interval_var.get()
                    
                    # Update monitoring enabled/disabled
                    self.health_monitor.monitoring_enabled = monitoring_var.get()
                    
                    messagebox.showinfo("Success", "Health monitoring configuration updated!")
                    config_dialog.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save configuration: {e}")
            
            tk.Button(
                button_frame,
                text="💾 Save",
                command=save_config,
                bg='#27AE60',
                fg='white',
                font=('Arial', 10)
            ).pack(side='right', padx=5)
            
            tk.Button(
                button_frame,
                text="❌ Cancel",
                command=config_dialog.destroy,
                bg='#E74C3C',
                fg='white',
                font=('Arial', 10)
            ).pack(side='right', padx=5)
            
        except Exception as e:
            self.logger.error(f"Failed to configure health monitoring: {e}")
            messagebox.showerror("Error", f"Failed to configure health monitoring: {e}")
    
    def setup_ai_coach_interface(self):
        """Setup the AI Coach chat interface."""
        try:
            # AI Coach Chat Frame
            coach_frame = tk.LabelFrame(
                self.root,
                text="🤖 AI COACH - Socratic Teaching & Strategic Questions",
                font=('Arial', 10, 'bold'),
                fg='#E67E22',  # Orange for the coach
                bg='#2C3E50'
            )
            coach_frame.pack(fill='x', padx=10, pady=5)
            
            # Check if coach is available
            if not hasattr(self, 'coach_available') or not self.coach_available:
                # Show unavailable message
                unavailable_label = tk.Label(
                    coach_frame,
                    text="⚠️ AI Coach not available - Check console for errors",
                    fg='#E74C3C',
                    bg='#2C3E50',
                    font=('Arial', 9)
                )
                unavailable_label.pack(pady=10)
                return
            
            # Chat display area
            chat_display_frame = tk.Frame(coach_frame, bg='#2C3E50')
            chat_display_frame.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Chat history text widget with scrollbar
            chat_scroll_frame = tk.Frame(chat_display_frame, bg='#2C3E50')
            chat_scroll_frame.pack(fill='both', expand=True)
            
            self.chat_display = tk.Text(
                chat_scroll_frame,
                height=8,
                bg='#34495E',
                fg='#ECF0F1',
                font=('Arial', 9),
                wrap=tk.WORD,
                state='disabled'  # Read-only
            )
            
            chat_scrollbar = tk.Scrollbar(chat_scroll_frame, command=self.chat_display.yview)
            self.chat_display.config(yscrollcommand=chat_scrollbar.set)
            
            self.chat_display.pack(side='left', fill='both', expand=True)
            chat_scrollbar.pack(side='right', fill='y')
            
            # Chat input frame
            input_frame = tk.Frame(coach_frame, bg='#2C3E50')
            input_frame.pack(fill='x', padx=5, pady=5)
            
            # Chat input field
            self.chat_input = tk.Entry(
                input_frame,
                bg='#34495E',
                fg='#ECF0F1',
                font=('Arial', 10),
                insertbackground='#ECF0F1'
            )
            self.chat_input.pack(side='left', fill='x', expand=True, padx=(0, 5))
            
            # Send button
            send_button = tk.Button(
                input_frame,
                text="💬 Ask Coach",
                font=('Arial', 9, 'bold'),
                bg='#E67E22',
                fg='white',
                command=self.send_chat_message,
                padx=10
            )
            send_button.pack(side='right')
            
            # Bind Enter key to send message
            self.chat_input.bind('<Return>', lambda e: self.send_chat_message())
            
            # Quick question buttons
            quick_frame = tk.Frame(coach_frame, bg='#2C3E50')
            quick_frame.pack(fill='x', padx=5, pady=(0, 5))
            
            tk.Label(
                quick_frame,
                text="Quick Questions:",
                fg='#BDC3C7',
                bg='#2C3E50',
                font=('Arial', 8)
            ).pack(side='left')
            
            quick_questions = [
                ("Why this pick?", "Why should I pick this card?"),
                ("Curve help", "How is my mana curve looking?"),
                ("Archetype advice", "What archetype should I focus on?"),
                ("Meta tips", "Any meta considerations for this draft?")
            ]
            
            for button_text, question in quick_questions:
                btn = tk.Button(
                    quick_frame,
                    text=button_text,
                    font=('Arial', 7),
                    bg='#95A5A6',
                    fg='white',
                    command=lambda q=question: self.send_quick_question(q),
                    padx=5
                )
                btn.pack(side='left', padx=2)
            
            # Welcome message
            self.add_chat_message("Coach", "👋 Welcome! I'm your AI draft coach. Ask me about strategy, card choices, or any draft questions. I'll use Socratic questioning to help you learn!")
            
        except Exception as e:
            print(f"⚠️ Failed to setup AI coach interface: {e}")
    
    def send_chat_message(self):
        """Send a chat message to the AI coach."""
        try:
            if not hasattr(self, 'conversational_coach') or not self.conversational_coach:
                messagebox.showerror("Error", "AI Coach not available")
                return
            
            # Get user message
            user_message = self.chat_input.get().strip()
            if not user_message:
                return
            
            # Clear input
            self.chat_input.delete(0, tk.END)
            
            # Add user message to chat
            self.add_chat_message("You", user_message)
            
            # Prepare context for the coach
            context = self.get_current_draft_context()
            
            # Get coach response
            try:
                coach_response = self.conversational_coach.process_user_query(user_message, context)
                self.add_chat_message("Coach", coach_response)
            except Exception as e:
                self.add_chat_message("Coach", f"Sorry, I encountered an error: {e}")
                self.logger.error(f"Coach response error: {e}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send message: {e}")
    
    def send_quick_question(self, question):
        """Send a quick question to the coach."""
        self.chat_input.delete(0, tk.END)
        self.chat_input.insert(0, question)
        self.send_chat_message()
    
    def add_chat_message(self, sender, message):
        """Add a message to the chat display."""
        try:
            self.chat_display.config(state='normal')
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M")
            
            # Format message based on sender
            if sender == "You":
                formatted_message = f"[{timestamp}] 🧑 {sender}: {message}\n\n"
            else:  # Coach
                formatted_message = f"[{timestamp}] 🤖 {sender}: {message}\n\n"
            
            # Insert message
            self.chat_display.insert(tk.END, formatted_message)
            
            # Scroll to bottom
            self.chat_display.see(tk.END)
            
            # Make read-only again
            self.chat_display.config(state='disabled')
            
        except Exception as e:
            print(f"Error adding chat message: {e}")
    
    def get_current_draft_context(self):
        """Get current draft context for the coach."""
        try:
            context = {
                'draft_stage': 'unknown',
                'hero_class': getattr(self, 'current_hero', None),
                'in_draft': getattr(self, 'in_draft', False),
                'draft_picks_count': getattr(self, 'draft_picks_count', 0),
                'last_analysis': None
            }
            
            # Add last analysis result if available
            if hasattr(self, 'last_full_analysis_result') and self.last_full_analysis_result:
                context['last_analysis'] = self.last_full_analysis_result
                context['draft_stage'] = 'card_selection'
            
            # Add hero context if in draft
            if self.in_draft and self.current_hero:
                context['hero_class'] = self.current_hero
                context['draft_stage'] = 'hero_selected' if self.draft_picks_count == 0 else 'drafting_cards'
            
            # Add archetype if available from last analysis
            if (hasattr(self, 'last_full_analysis_result') and 
                self.last_full_analysis_result and 
                hasattr(self.last_full_analysis_result, 'deck_analysis')):
                deck_analysis = self.last_full_analysis_result.deck_analysis
                if isinstance(deck_analysis, dict) and 'archetype' in deck_analysis:
                    context['archetype'] = deck_analysis['archetype']
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting draft context: {e}")
            return {'draft_stage': 'unknown'}
    
    def setup_draft_progression_display(self):
        """Setup the draft progression display showing Hero → Cards relationship."""
        # Main draft progression frame
        self.progression_frame = tk.LabelFrame(
            self.root,
            text="🎯 DRAFT PROGRESSION - AI v2 SYSTEM",
            font=('Arial', 11, 'bold'),
            fg='#E74C3C',  # Red for prominence
            bg='#2C3E50'
        )
        self.progression_frame.pack(fill='x', padx=10, pady=5)
        
        # Create main progression container
        progression_container = tk.Frame(self.progression_frame, bg='#2C3E50')
        progression_container.pack(fill='x', padx=10, pady=10)
        
        # Hero Selection Section
        hero_section = tk.Frame(progression_container, bg='#34495E', relief='raised', bd=2)
        hero_section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        tk.Label(
            hero_section,
            text="👑 HERO SELECTION",
            font=('Arial', 10, 'bold'),
            fg='#F39C12',
            bg='#34495E'
        ).pack(pady=5)
        
        # Hero display area
        self.hero_display_frame = tk.Frame(hero_section, bg='#34495E')
        self.hero_display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.hero_status_label = tk.Label(
            self.hero_display_frame,
            text="⏳ Waiting for hero selection...",
            font=('Arial', 9),
            fg='#BDC3C7',
            bg='#34495E',
            wraplength=180
        )
        self.hero_status_label.pack(pady=10)
        
        # Hero winrate display (initially hidden)
        self.hero_winrate_frame = tk.Frame(self.hero_display_frame, bg='#34495E')
        self.hero_winrate_label = tk.Label(
            self.hero_winrate_frame,
            text="",
            font=('Arial', 8),
            fg='#27AE60',
            bg='#34495E'
        )
        self.hero_winrate_label.pack()
        
        # Progression Arrow
        arrow_frame = tk.Frame(progression_container, bg='#2C3E50')
        arrow_frame.pack(side='left', padx=10, pady=5)
        
        tk.Label(
            arrow_frame,
            text="➤",
            font=('Arial', 20, 'bold'),
            fg='#3498DB',
            bg='#2C3E50'
        ).pack(pady=20)
        
        # Card Draft Section
        card_section = tk.Frame(progression_container, bg='#34495E', relief='raised', bd=2)
        card_section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        tk.Label(
            card_section,
            text="🃏 CARD DRAFTING",
            font=('Arial', 10, 'bold'),
            fg='#3498DB',
            bg='#34495E'
        ).pack(pady=5)
        
        # Card draft status
        self.card_draft_frame = tk.Frame(card_section, bg='#34495E')
        self.card_draft_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.card_status_label = tk.Label(
            self.card_draft_frame,
            text="⏳ Waiting for hero selection...",
            font=('Arial', 9),
            fg='#BDC3C7',
            bg='#34495E',
            wraplength=180
        )
        self.card_status_label.pack(pady=10)
        
        # Card pick counter
        self.pick_counter_label = tk.Label(
            self.card_draft_frame,
            text="Pick: 0/30",
            font=('Arial', 8, 'bold'),
            fg='#9B59B6',
            bg='#34495E'
        )
        self.pick_counter_label.pack()
        
        # Draft Phase Indicator
        phase_frame = tk.Frame(progression_container, bg='#2C3E50')
        phase_frame.pack(side='left', padx=10, pady=5)
        
        tk.Label(
            phase_frame,
            text="📈 PHASE",
            font=('Arial', 9, 'bold'),
            fg='#ECF0F1',
            bg='#2C3E50'
        ).pack()
        
        self.phase_indicator_label = tk.Label(
            phase_frame,
            text="WAITING",
            font=('Arial', 8, 'bold'),
            fg='#95A5A6',
            bg='#2C3E50',
            relief='sunken',
            bd=1,
            padx=5,
            pady=2
        )
        self.phase_indicator_label.pack(pady=5)
        
        # System Status Section
        status_section = tk.Frame(progression_container, bg='#34495E', relief='raised', bd=2)
        status_section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        tk.Label(
            status_section,
            text="🔧 DATA SOURCES",
            font=('Arial', 10, 'bold'),
            fg='#E67E22',
            bg='#34495E'
        ).pack(pady=5)
        
        self.ai_status_frame = tk.Frame(status_section, bg='#34495E')
        self.ai_status_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # HSReplay Heroes indicator
        self.hsreplay_heroes_label = tk.Label(
            self.ai_status_frame,
            text="🔄 HSReplay Heroes",
            font=('Arial', 7),
            fg='#F39C12',
            bg='#34495E'
        )
        self.hsreplay_heroes_label.pack(anchor='w')
        
        # HSReplay Cards indicator  
        self.hsreplay_cards_label = tk.Label(
            self.ai_status_frame,
            text="🔄 HSReplay Cards",
            font=('Arial', 7),
            fg='#F39C12',
            bg='#34495E'
        )
        self.hsreplay_cards_label.pack(anchor='w')
        
        # HearthArena fallback indicator
        self.heartharena_label = tk.Label(
            self.ai_status_frame,
            text="🔄 HearthArena",
            font=('Arial', 7),
            fg='#F39C12',
            bg='#34495E'
        )
        self.heartharena_label.pack(anchor='w')
        
        # Fallback mode indicator
        self.fallback_mode_label = tk.Label(
            self.ai_status_frame,
            text="",
            font=('Arial', 7, 'bold'),
            fg='#E74C3C',
            bg='#34495E'
        )
        self.fallback_mode_label.pack(anchor='w')
        
        # Initialize display
        self.update_draft_progression_display()
    
    def update_draft_progression_display(self):
        """Update the draft progression display with current state."""
        try:
            # Get current phase
            current_phase = getattr(self, 'current_draft_phase', 'waiting')
            selected_hero = getattr(self, 'selected_hero_class', None)
            pick_count = getattr(self, 'draft_picks_count', 0)
            
            # Update phase indicator
            phase_colors = {
                'waiting': ('#95A5A6', 'WAITING'),
                'hero_selection': ('#F39C12', 'HERO SELECT'),
                'card_picks': ('#3498DB', 'CARD PICKS'),
                'complete': ('#27AE60', 'COMPLETE')
            }
            
            color, text = phase_colors.get(current_phase, ('#95A5A6', 'UNKNOWN'))
            self.phase_indicator_label.config(text=text, fg=color)
            
            # Update hero section
            if current_phase == 'waiting':
                self.hero_status_label.config(text="⏳ Waiting for hero selection...")
                self.hero_winrate_frame.pack_forget()
            elif current_phase == 'hero_selection':
                self.hero_status_label.config(text="👑 Analyzing hero options...", fg='#F39C12')
            elif selected_hero:
                self.hero_status_label.config(text=f"👑 Selected: {selected_hero}", fg='#27AE60')
                
                # Show hero winrate if available
                if hasattr(self, 'hero_recommendation') and self.hero_recommendation:
                    winrates = self.hero_recommendation.winrates
                    if selected_hero in winrates:
                        winrate = winrates[selected_hero]
                        confidence = self.hero_recommendation.confidence_level
                        self.hero_winrate_label.config(
                            text=f"Winrate: {winrate:.1f}% | Confidence: {confidence:.1%}"
                        )
                        self.hero_winrate_frame.pack(fill='x')
            
            # Update card section
            if current_phase == 'waiting':
                self.card_status_label.config(text="⏳ Waiting for hero selection...")
            elif current_phase == 'hero_selection':
                self.card_status_label.config(text="⏳ Preparing for card picks...", fg='#F39C12')
            elif current_phase == 'card_picks':
                if selected_hero:
                    self.card_status_label.config(
                        text=f"🎯 Hero-aware recommendations\nfor {selected_hero}",
                        fg='#3498DB'
                    )
                else:
                    self.card_status_label.config(text="🃏 Card recommendations ready", fg='#3498DB')
            elif current_phase == 'complete':
                self.card_status_label.config(text="✅ Draft complete!", fg='#27AE60')
            
            # Update pick counter
            self.pick_counter_label.config(text=f"Pick: {pick_count}/30")
            
            # Update AI status based on system health
            self.update_ai_status_display()
            
        except Exception as e:
            print(f"Error updating draft progression display: {e}")
    
    def update_ai_status_display(self):
        """Update comprehensive data source indicators with fallback modes."""
        try:
            if not hasattr(self, 'ai_v2_integrator') or not self.ai_v2_integrator:
                self._set_all_data_sources_offline()
                return
            
            # Get comprehensive system health
            health = self.ai_v2_integrator.check_system_health()
            data_sources = health.get('data_sources', {})
            components = health.get('components', {})
            
            # Update HSReplay Heroes status
            hero_status = self._get_hero_data_status(data_sources, components)
            self._update_data_source_indicator(self.hsreplay_heroes_label, "HSReplay Heroes", hero_status)
            
            # Update HSReplay Cards status
            card_status = self._get_card_data_status(data_sources, components)
            self._update_data_source_indicator(self.hsreplay_cards_label, "HSReplay Cards", card_status)
            
            # Update HearthArena fallback status
            heartharena_status = self._get_heartharena_status()
            self._update_data_source_indicator(self.heartharena_label, "HearthArena", heartharena_status)
            
            # Update fallback mode indicator
            self._update_fallback_mode_indicator(hero_status, card_status, heartharena_status)
            
        except Exception as e:
            self._set_all_data_sources_error()
    
    def _get_hero_data_status(self, data_sources, components):
        """Get hero data source status with cache age info."""
        try:
            hsreplay_data = data_sources.get('hsreplay', {})
            hero_selector_data = components.get('hero_selector', {})
            
            if hsreplay_data.get('status') == 'online':
                cache_age = hsreplay_data.get('hero_cache_age_hours', 0)
                if cache_age < 12:  # Fresh data
                    return {'status': 'online', 'detail': f'Fresh ({cache_age:.1f}h)'}
                elif cache_age < 24:  # Aging data
                    return {'status': 'aging', 'detail': f'Aging ({cache_age:.1f}h)'}
                else:  # Stale data
                    return {'status': 'stale', 'detail': f'Stale ({cache_age:.1f}h)'}
            elif hero_selector_data.get('status') == 'degraded':
                return {'status': 'cached', 'detail': 'Cached only'}
            else:
                return {'status': 'offline', 'detail': 'Unavailable'}
        except:
            return {'status': 'error', 'detail': 'Error'}
    
    def _get_card_data_status(self, data_sources, components):
        """Get card data source status with cache age info."""
        try:
            hsreplay_data = data_sources.get('hsreplay', {})
            card_evaluator_data = components.get('card_evaluator', {})
            
            if hsreplay_data.get('status') == 'online':
                cache_age = hsreplay_data.get('card_cache_age_hours', 0)
                if cache_age < 24:  # Fresh data  
                    return {'status': 'online', 'detail': f'Fresh ({cache_age:.1f}h)'}
                elif cache_age < 48:  # Aging data
                    return {'status': 'aging', 'detail': f'Aging ({cache_age:.1f}h)'}
                else:  # Stale data
                    return {'status': 'stale', 'detail': f'Stale ({cache_age:.1f}h)'}
            elif card_evaluator_data.get('status') == 'degraded':
                return {'status': 'cached', 'detail': 'Cached only'}
            else:
                return {'status': 'offline', 'detail': 'Unavailable'}
        except:
            return {'status': 'error', 'detail': 'Error'}
    
    def _get_heartharena_status(self):
        """Get HearthArena fallback status."""
        try:
            # Check if advisor is available (legacy system)
            if hasattr(self, 'advisor') and self.advisor:
                return {'status': 'online', 'detail': 'Available'}
            else:
                return {'status': 'offline', 'detail': 'Unavailable'}
        except:
            return {'status': 'error', 'detail': 'Error'}
    
    def _update_data_source_indicator(self, label_widget, source_name, status_info):
        """Update individual data source indicator."""
        status = status_info['status']
        detail = status_info['detail']
        
        # Status icons and colors
        status_config = {
            'online': ('✅', '#27AE60', ''),
            'aging': ('⚠️', '#F39C12', ' (Aging)'),
            'stale': ('⚠️', '#E67E22', ' (Stale)'),
            'cached': ('💾', '#9B59B6', ' (Cached)'),
            'offline': ('❌', '#E74C3C', ' (Offline)'),
            'error': ('💥', '#E74C3C', ' (Error)')
        }
        
        icon, color, suffix = status_config.get(status, ('❓', '#95A5A6', ' (Unknown)'))
        display_text = f"{icon} {source_name}{suffix}"
        
        label_widget.config(text=display_text, fg=color)
    
    def _update_fallback_mode_indicator(self, hero_status, card_status, heartharena_status):
        """Update fallback mode indicator based on data source availability."""
        hero_online = hero_status['status'] == 'online'
        card_online = card_status['status'] == 'online'
        heartharena_online = heartharena_status['status'] == 'online'
        
        if hero_online and card_online:
            # Full AI v2 mode
            self.fallback_mode_label.config(text="🎯 Full AI v2 Mode", fg='#27AE60')
        elif (hero_status['status'] in ['aging', 'cached']) and (card_status['status'] in ['aging', 'cached']):
            # Degraded AI v2 mode
            self.fallback_mode_label.config(text="⚠️ Degraded AI v2", fg='#F39C12')
        elif heartharena_online:
            # HearthArena fallback mode
            self.fallback_mode_label.config(text="💾 HearthArena Mode", fg='#9B59B6')
        else:
            # Emergency fallback mode
            self.fallback_mode_label.config(text="🚨 Emergency Mode", fg='#E74C3C')
    
    def _set_all_data_sources_offline(self):
        """Set all data source indicators to offline."""
        offline_config = ("❌", "#E74C3C")
        self.hsreplay_heroes_label.config(text="❌ HSReplay Heroes (Offline)", fg=offline_config[1])
        self.hsreplay_cards_label.config(text="❌ HSReplay Cards (Offline)", fg=offline_config[1])
        self.heartharena_label.config(text="❌ HearthArena (Offline)", fg=offline_config[1])
        self.fallback_mode_label.config(text="❌ AI v2 Offline", fg=offline_config[1])
    
    def _set_all_data_sources_error(self):
        """Set all data source indicators to error state."""
        error_config = ("💥", "#E74C3C")
        self.hsreplay_heroes_label.config(text="💥 HSReplay Heroes (Error)", fg=error_config[1])
        self.hsreplay_cards_label.config(text="💥 HSReplay Cards (Error)", fg=error_config[1])
        self.heartharena_label.config(text="💥 HearthArena (Error)", fg=error_config[1])
        self.fallback_mode_label.config(text="💥 System Error", fg=error_config[1])
    
    def toggle_debug_mode(self, *args):
        """Toggle debug mode on/off with immediate and accurate feedback."""
        # The .get() method retrieves the CURRENT state of the checkbox
        if self.debug_enabled.get():
            enable_debug()
            self.log_text("🐛 DEBUG MODE ENABLED - Visual debugging and detailed metrics are now active.")
        else:
            disable_debug()
            self.log_text("📊 DEBUG MODE DISABLED - Running in normal operation mode.")
    
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

    def run_validation_suite(self):
        """Run the comprehensive validation suite."""
        self.log_text("🧪 Starting comprehensive validation suite...")
        
        # Disable the button during validation
        self.validation_btn.configure(state='disabled', text="🧪 RUNNING...")
        
        def validation_thread():
            try:
                # Check system health first
                if not check_system_health():
                    self.log_text("❌ System health check failed - validation aborted")
                    return
                
                self.log_text("✅ System health check passed")
                self.log_text("🔍 Running full validation suite with all detection methods...")
                
                # Run full validation
                results = run_full_validation()
                
                # Log summary results
                overall = results.get('overall_scores', {})
                self.log_text(f"🏆 Best Method: {overall.get('best_method', 'Unknown')}")
                self.log_text(f"📊 Average IoU: {overall.get('average_iou', 0):.3f}")
                self.log_text(f"⏱️ Average Timing: {overall.get('average_timing', 0):.1f}ms")
                self.log_text(f"✅ Overall Pass Rate: {overall.get('overall_pass_rate', 0):.1%}")
                
                # Show recommendations
                recommendations = results.get('recommendations', [])
                if recommendations:
                    self.log_text("💡 RECOMMENDATIONS:")
                    for rec in recommendations:
                        self.log_text(f"   {rec}")
                
                # Show detailed results in popup window
                self._show_validation_results_window(results)
                
            except Exception as e:
                self.log_text(f"❌ Validation suite failed: {e}")
                self.logger.error(f"Validation suite error: {e}")
            finally:
                # Re-enable the button
                self.validation_btn.configure(state='normal', text="🧪 VALIDATE")
        
        # Run validation in background thread
        threading.Thread(target=validation_thread, daemon=True).start()
    
    def _show_validation_results_window(self, results):
        """Display validation results in a dedicated window."""
        results_window = tk.Toplevel(self.root)
        results_window.title("🧪 Validation Suite Results")
        results_window.geometry("900x700")
        results_window.configure(bg='#2C3E50')
        
        # Create notebook for tabbed view
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Summary tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📊 Summary")
        
        summary_text = scrolledtext.ScrolledText(
            summary_frame,
            bg='#1C1C1C',
            fg='#ECF0F1',
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        summary_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Format summary content
        summary_content = self._format_validation_summary(results)
        summary_text.insert(tk.END, summary_content)
        summary_text.configure(state='disabled')
        
        # Method details tab
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="🔬 Method Details")
        
        details_text = scrolledtext.ScrolledText(
            details_frame,
            bg='#1C1C1C',
            fg='#ECF0F1',
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        details_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Format method details
        details_content = self._format_validation_details(results)
        details_text.insert(tk.END, details_content)
        details_text.configure(state='disabled')
    
    def _format_validation_summary(self, results):
        """Format validation summary for display."""
        text = "🧪 ARENA BOT VALIDATION SUITE RESULTS\n"
        text += "=" * 80 + "\n\n"
        
        # Overall scores
        overall = results.get('overall_scores', {})
        text += f"🏆 Best Method: {overall.get('best_method', 'Unknown')}\n"
        text += f"📊 Average IoU: {overall.get('average_iou', 0):.3f}\n"
        text += f"⏱️ Average Timing: {overall.get('average_timing', 0):.1f}ms\n"
        text += f"✅ Overall Pass Rate: {overall.get('overall_pass_rate', 0):.1%}\n\n"
        
        # Method summary
        text += "📋 METHOD SUMMARY:\n"
        for method, results_data in results.get('method_results', {}).items():
            if results_data['tests_run'] > 0:
                pass_fail = results.get('pass_fail_summary', {}).get(method, {})
                status = "✅ PASS" if pass_fail.get('final_result') else "❌ FAIL"
                text += f"   {method}: {status} "
                text += f"(IoU: {results_data.get('avg_iou', 0):.3f}, "
                text += f"Time: {results_data.get('avg_timing', 0):.1f}ms)\n"
        
        # Recommendations
        text += "\n💡 RECOMMENDATIONS:\n"
        for rec in results.get('recommendations', []):
            text += f"   {rec}\n"
        
        return text
    
    def _format_validation_details(self, results):
        """Format detailed validation results for display."""
        text = "🔬 DETAILED METHOD RESULTS\n"
        text += "=" * 80 + "\n\n"
        
        for method, method_data in results.get('method_results', {}).items():
            if method_data['tests_run'] == 0:
                continue
                
            text += f"📊 {method.upper()}:\n"
            text += f"   Tests Run: {method_data['tests_run']}\n"
            text += f"   Tests Passed: {method_data['tests_passed']}\n"
            text += f"   Pass Rate: {method_data.get('pass_rate', 0):.1%}\n"
            text += f"   Average IoU: {method_data.get('avg_iou', 0):.3f}\n"
            text += f"   Average Timing: {method_data.get('avg_timing', 0):.1f}ms\n"
            
            # Grade distribution
            grades = method_data.get('grade_distribution', {})
            text += f"   Grade Distribution: "
            for grade in ['A', 'B', 'C', 'D', 'F']:
                text += f"{grade}:{grades.get(grade, 0)} "
            text += "\n\n"
        
        # Performance results
        perf_results = results.get('performance_results', {})
        if perf_results:
            text += "⚡ PERFORMANCE BENCHMARK:\n"
            for method, perf_data in perf_results.items():
                text += f"   {method}:\n"
                text += f"      Avg Time: {perf_data.get('avg_time_ms', 0):.1f}ms\n"
                text += f"      Min Time: {perf_data.get('min_time_ms', 0):.1f}ms\n"
                text += f"      Max Time: {perf_data.get('max_time_ms', 0):.1f}ms\n"
                text += f"      Meets Threshold: {perf_data.get('meets_threshold', False)}\n\n"
        
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
            
            # Create and show the overlay
            if not self.overlay:
                try:
                    self.overlay = DraftOverlay()
                    threading.Thread(target=self.overlay.start, daemon=True).start()
                    self.log_text("✅ Visual overlay started.")
                except Exception as e:
                    self.log_text(f"⚠️ Could not start visual overlay: {e}")
                    self.overlay = None

            # Start log monitoring if available
            if self.log_monitor:
                self.log_monitor.start_monitoring()
                self.log_text("✅ Started log monitoring")
            
            self.log_text("🎯 Monitoring started - waiting for Arena drafts...")
        else:
            self.running = False
            self.start_btn.config(text="▶️ START MONITORING", bg='#27AE60')
            self.update_status("Monitoring Stopped")
            
            # Stop and destroy the overlay
            if self.overlay:
                self.overlay.stop()
                self.overlay = None
                self.log_text("✅ Visual overlay stopped.")
            
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
    
    def open_advanced_correction_center(self):
        """Open the advanced manual correction center for both hero and card corrections."""
        self.log_text("🔧 Advanced manual correction center opened.")
        messagebox.showinfo("Coming Soon", "The advanced manual correction center is under development.")
    
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
    
    def open_advanced_correction_center(self):
        """Open the advanced manual correction center for both hero and card corrections."""
        self.log_text("🔧 Advanced manual correction center opened.")
        messagebox.showinfo("Coming Soon", "The advanced manual correction center is under development.")
    
    def open_advanced_correction_center_simple(self):
        """Simple fallback method for advanced correction center."""
        self.log_text("🔧 Advanced manual correction center opened.")
        messagebox.showinfo("Coming Soon", "The advanced manual correction center is under development.")
    
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
                    'logs': [f"⚠️ PIL ImageGrab failed: {e}"]
                })
                return
            
            if screenshot is not None:
                # This is the potentially slow part - analyze the screenshot
                analysis_result = self.analyze_screenshot_data(screenshot)
                
                # The analysis result now contains success, logs, and data
                # Add screenshot capture log to the logs
                if 'logs' in analysis_result:
                    analysis_result['logs'].insert(0, "✅ Screenshot captured with PIL ImageGrab")
                
                # Put the result in the queue for the main thread
                self.result_queue.put({
                    'success': analysis_result.get('success', False),
                    'result': analysis_result,
                    'logs': analysis_result.get('logs', []),
                    'error': analysis_result.get('error', None)
                })
            else:
                self.result_queue.put({
                    'success': False,
                    'error': 'Could not take screenshot',
                    'logs': ["❌ Could not take screenshot"]
                })
                
        except Exception as e:
            # Put any errors in the queue
            self.result_queue.put({
                'success': False,
                'error': str(e),
                'logs': [f"❌ Screenshot analysis failed: {e}"]
            })
    
    def _check_for_result(self):
        """Check the result queue and update the UI when analysis is complete."""
        try:
            queue_result = self.result_queue.get_nowait()

            # Log messages returned from the thread
            if 'logs' in queue_result and queue_result['logs']:
                for log_msg in queue_result['logs']:
                    self.log_text(log_msg)
            
            if queue_result.get('success'):
                result = queue_result.get('result')
                if result:
                    self.last_full_analysis_result = result
                    self.show_analysis_result(result)
                    # --- FIX: Update the overlay with the results ---
                    if self.overlay:
                        self.overlay.optimized_update_display(result)
                else:
                    self.log_text("❌ Analysis returned no result.")
            else:
                error = queue_result.get('error', 'Unknown error')
                self.log_text(f"❌ Analysis failed: {error}")
            
            self.analysis_in_progress = False
            self.screenshot_btn.config(state=tk.NORMAL)
            self.update_status("Analysis complete. Ready for next screenshot.")
            self.progress_bar.stop()
            self.progress_frame.pack_forget()
            
        except:
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
        """
        Analyzes screenshot data in a thread-safe way, returning results and logs.
        This method is designed to be run in a background thread.
        """
        logs = []
        if screenshot is None:
            return {'success': False, 'error': 'Screenshot capture failed.', 'logs': logs}

        height, width = screenshot.shape[:2]
        resolution_str = f"{width}x{height}"
        logs.append(f"🔍 Analyzing screenshot: {resolution_str}")

        # --- DEBUG LOGGING BLOCK ---
        if is_debug_enabled():
            logs.append(f"🐛 DEBUG: Debug mode is active.")
            if self.use_custom_coords.get() and self.custom_coordinates:
                logs.append(f"🐛 DEBUG: Using custom coordinates: {self.custom_coordinates}")
            else:
                logs.append(f"🐛 DEBUG: Using automatic coordinate detection.")
            logs.append(f"🐛 DEBUG: Ultimate Detection toggle is {'ON' if self.use_ultimate_detection.get() else 'OFF'}.")
        # --- END OF DEBUG LOGGING BLOCK ---

        card_regions = None
        
        if self.smart_detector:
            try:
                smart_result = self.smart_detector.detect_cards_automatically(screenshot)
                if smart_result and smart_result.get('success'):
                    card_regions = smart_result.get('card_positions', [])
                    logs.append(f"✅ Smart detector found {len(card_regions)} regions.")
                else:
                    logs.append("⚠️ Smart detector failed, using fallback.")
            except Exception as e:
                logs.append(f"⚠️ Smart detector error: {e}")
        
        if not card_regions:
            logs.append("📐 Using resolution-based coordinate fallback.")
            card_regions = [
                (1100, 75, 250, 350), (1375, 75, 250, 350), (1650, 75, 250, 350)
            ]

        detected_cards = []
        if self.histogram_matcher:
            for i, (x, y, w, h) in enumerate(card_regions):
                if (y + h <= height and x + w <= width and x >= 0 and y >= 0):
                    card_region = screenshot[y:y+h, x:x+w]
                    match = self.histogram_matcher.match_card(card_region)
                    if match:
                        detected_cards.append({
                            'card_code': match.card_code,
                            'card_name': self.get_card_name(match.card_code),
                            'confidence': match.confidence,
                            'region': (x,y,w,h)
                        })
        
        recommendation = None
        if self.advisor and len(detected_cards) >= 3:
            card_codes = [c['card_code'] for c in detected_cards]
            try:
                choice = self.advisor.analyze_draft_choice(card_codes, self.current_hero or 'unknown')
                recommendation = {
                    'recommended_pick': choice.recommended_pick + 1,
                    'recommended_card': choice.cards[choice.recommended_pick].card_code,
                    'reasoning': choice.reasoning,
                    'card_details': [vars(c) for c in choice.cards]
                }
            except Exception as e:
                logs.append(f"⚠️ AI recommendation failed: {e}")

        # Visual debugging if enabled
        if is_debug_enabled() and card_regions:
            try:
                # Create debug visualization
                detected_boxes = [card['region'] for card in detected_cards]
                card_names = [card['card_name'] for card in detected_cards]
                confidences = [card['confidence'] for card in detected_cards]
                
                debug_img = create_debug_visualization(
                    screenshot,
                    detected_boxes,
                    ground_truth_boxes=None,  # No ground truth in live detection
                    detection_method="live_detection",
                    card_names=card_names,
                    confidences=confidences
                )
                
                # Save debug image
                debug_path = save_debug_image(debug_img, "live_analysis", "live_detection")
                if debug_path:
                    logs.append(f"🐛 Debug image saved: {debug_path}")
                
                # Log metrics if available
                log_detection_metrics(
                    screenshot_file="live_capture",
                    resolution=resolution_str,
                    method="live_detection",
                    detected_boxes=detected_boxes,
                    ground_truth_boxes=None,
                    card_names=card_names,
                    confidences=confidences
                )
                
            except Exception as e:
                logs.append(f"⚠️ Debug visualization failed: {e}")

        return {
            'success': len(detected_cards) > 0,
            'detected_cards': detected_cards,
            'recommendation': recommendation,
            'logs': logs
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
        """Enhanced analysis result display with hero context and AI v2 integration."""
        detected_cards = result['detected_cards']
        recommendation = result['recommendation']
        
        # Sort detected cards by x-coordinate to ensure correct UI positioning (left to right)
        detected_cards_sorted = sorted(detected_cards, key=lambda c: c['region'][0] if 'region' in c else 0)
        self.log_text(f"   🔄 Sorted {len(detected_cards_sorted)} cards by screen position (left to right)")
        
        # Update card images in GUI with sorted cards
        self.update_card_images(detected_cards_sorted)
        
        # Enhanced logging with hero context
        hero_context = f" for {self.selected_hero_class}" if self.selected_hero_class else ""
        phase_context = f" [{self.current_draft_phase}]" if hasattr(self, 'current_draft_phase') else ""
        
        self.log_text(f"\n✅ Detected {len(detected_cards_sorted)} cards{hero_context}{phase_context}:")
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
        
        # Enhanced recommendation display with hero context
        if recommendation:
            self._show_enhanced_recommendation(recommendation, detected_cards_sorted)
        else:
            # Try AI v2 recommendation if legacy system failed
            if self.current_draft_phase == "card_picks" and self.selected_hero_class:
                self._try_ai_v2_recommendation(detected_cards_sorted)
            else:
                self.show_recommendation("Cards Detected", f"Found {len(detected_cards)} cards but no AI recommendation available.")
    
    def _show_enhanced_recommendation(self, recommendation, detected_cards):
        """Show enhanced recommendation with hero context and AI v2 integration."""
        try:
            rec_card_name = self.get_card_name(recommendation['recommended_card'])
            
            # Enhanced title with hero context
            hero_context = f" for {self.selected_hero_class}" if self.selected_hero_class else ""
            pick_context = f" (Pick #{self.draft_picks_count + 1})" if hasattr(self, 'draft_picks_count') else ""
            
            rec_text = f"🎯 AI v2 RECOMMENDATION{hero_context}{pick_context}\n"
            rec_text += f"👑 PICK: {rec_card_name}\n\n"
            rec_text += f"📊 Position: #{recommendation['recommended_pick']}\n"
            
            # Add hero synergy context if available
            if self.selected_hero_class:
                rec_text += f"🎯 Hero: {self.selected_hero_class}\n"
                rec_text += f"📈 Phase: {getattr(self, 'current_draft_phase', 'card_picks')}\n"
            
            rec_text += f"\n💭 Reasoning: {recommendation['reasoning']}\n\n"
            rec_text += "📋 All Options:\n"
            
            for i, card_detail in enumerate(recommendation['card_details']):
                card_name = self.get_card_name(card_detail['card_code'])
                marker = "👑" if i == recommendation['recommended_pick'] - 1 else "📋"
                
                # Enhanced card details with tier and winrate
                tier = card_detail.get('tier_letter', 'N/A')
                winrate = card_detail.get('win_rate', 0.0)
                rec_text += f"{marker} {i+1}. {card_name} (Tier {tier}, {winrate:.0%} WR)\n"
                
                # Enhanced hero synergy analysis
                if self.selected_hero_class:
                    synergy_analysis = self._analyze_hero_card_synergy(card_detail['card_code'], self.selected_hero_class)
                    if synergy_analysis:
                        rec_text += f"     {synergy_analysis}\n"
            
            # Show AI v2 confidence if available
            if hasattr(self, 'hero_recommendation') and self.hero_recommendation:
                hero_confidence = self.hero_recommendation.confidence_level
                rec_text += f"\n🔧 System Confidence: Hero {hero_confidence:.1%} | Cards: Variable\n"
            
            self.show_recommendation("AI v2 Draft Recommendation", rec_text)
            self.log_text(f"\n🎯 AI v2 RECOMMENDATION{hero_context}: Pick #{recommendation['recommended_pick']} - {rec_card_name}")
            
        except Exception as e:
            self.log_text(f"⚠️ Error displaying enhanced recommendation: {e}")
            # Fallback to original display
            self._show_basic_recommendation(recommendation)
    
    def _try_ai_v2_recommendation(self, detected_cards):
        """Try to get AI v2 recommendation when legacy system fails."""
        try:
            if not self.ai_v2_integrator or not self.selected_hero_class:
                self.show_recommendation("Cards Detected", f"Found {len(detected_cards)} cards but no AI recommendation available.")
                return
            
            # Extract card IDs from detected cards
            card_ids = [card.get('card_code', '') for card in detected_cards if card.get('card_code')]
            
            if len(card_ids) != 3:
                self.log_text(f"⚠️ AI v2 needs exactly 3 cards, found {len(card_ids)}")
                return
            
            # Create deck state for AI v2
            from arena_bot.ai_v2.data_models import DeckState
            deck_state = DeckState(
                hero_class=self.selected_hero_class,
                archetype="Balanced",  # Default archetype for now
                drafted_cards=[],  # TODO: Track drafted cards
                pick_number=getattr(self, 'draft_picks_count', 0) + 1
            )
            
            self.log_text("🎯 Generating AI v2 fallback recommendation...")
            ai_decision = self.ai_v2_integrator.get_ai_decision_with_recovery(deck_state, card_ids)
            
            # Convert AI v2 decision to legacy format for display
            legacy_recommendation = self._convert_ai_v2_to_legacy(ai_decision, detected_cards)
            self._show_enhanced_recommendation(legacy_recommendation, detected_cards)
            
        except Exception as e:
            self.log_text(f"⚠️ AI v2 fallback failed: {e}")
            self.show_recommendation("Cards Detected", f"Found {len(detected_cards)} cards but no AI recommendation available.")
    
    def _convert_ai_v2_to_legacy(self, ai_decision, detected_cards):
        """Convert AI v2 decision format to legacy recommendation format."""
        try:
            recommended_index = ai_decision.recommended_pick_index
            recommended_card = detected_cards[recommended_index].get('card_code', '')
            
            card_details = []
            for i, card in enumerate(detected_cards):
                card_analysis = ai_decision.all_offered_cards_analysis[i] if i < len(ai_decision.all_offered_cards_analysis) else {}
                
                card_details.append({
                    'card_code': card.get('card_code', ''),
                    'tier_letter': 'AI',  # AI v2 doesn't use tier letters
                    'win_rate': card_analysis.get('scores', {}).get('final_score', 0.5)
                })
            
            return {
                'recommended_card': recommended_card,
                'recommended_pick': recommended_index + 1,
                'reasoning': ai_decision.comparative_explanation,
                'card_details': card_details
            }
            
        except Exception as e:
            self.log_text(f"⚠️ Error converting AI v2 decision: {e}")
            return None
    
    def _show_basic_recommendation(self, recommendation):
        """Fallback to basic recommendation display."""
        rec_card_name = self.get_card_name(recommendation['recommended_card'])
        rec_text = f"🎯 RECOMMENDED PICK: {rec_card_name}\n\n"
        rec_text += f"📊 Position: #{recommendation['recommended_pick']}\n\n"
        rec_text += f"💭 Reasoning: {recommendation['reasoning']}\n\n"
        rec_text += "📋 All Cards:\n"
        
        for i, card_detail in enumerate(recommendation['card_details']):
            card_name = self.get_card_name(card_detail['card_code'])
            marker = "👑" if i == recommendation['recommended_pick'] - 1 else "📋"
            rec_text += f"{marker} {i+1}. {card_name} (Tier {card_detail['tier_letter']}, {card_detail['win_rate']:.0%} WR)\n"
        
        self.show_recommendation("AI Draft Recommendation", rec_text)
    
    def _get_card_class_context(self, card_code):
        """Get card class for hero synergy context."""
        try:
            if hasattr(self, 'cards_loader') and self.cards_loader:
                return self.cards_loader.get_card_class(card_code)
            return 'UNKNOWN'
        except:
            return 'UNKNOWN'
    
    def _analyze_hero_card_synergy(self, card_code, hero_class):
        """Analyze and explain hero-card synergy with detailed explanations."""
        try:
            # Get card properties
            card_class = self._get_card_class_context(card_code)
            card_name = self.get_card_name(card_code) if hasattr(self, 'get_card_name') else card_code
            
            # Class synergy analysis
            if card_class == hero_class:
                return self._get_class_synergy_explanation(card_code, card_name, hero_class)
            elif card_class == 'NEUTRAL':
                return self._get_neutral_synergy_explanation(card_code, card_name, hero_class)
            else:
                return f"❓ Off-class card ({card_class})"
        except Exception as e:
            return None
    
    def _get_class_synergy_explanation(self, card_code, card_name, hero_class):
        """Get detailed class synergy explanation."""
        # Hero-specific class synergy patterns
        class_synergies = {
            'WARRIOR': {
                'weapon': '⚔️ Weapon synergy - enhances board control',
                'armor': '🛡️ Armor synergy - defensive value',
                'rush': '🏃 Rush synergy - immediate board impact',
                'taunt': '🛡️ Taunt synergy - control gameplan'
            },
            'MAGE': {
                'spell': '✨ Spell synergy - combo potential',
                'freeze': '🧊 Freeze synergy - tempo control', 
                'secret': '🤫 Secret synergy - board protection',
                'elemental': '🔥 Elemental synergy - spell power'
            },
            'HUNTER': {
                'beast': '🐺 Beast synergy - tribal value',
                'secret': '🏹 Secret synergy - aggressive pressure',
                'weapon': '🏹 Weapon synergy - face damage',
                'rush': '🏃 Rush synergy - board control'
            },
            'PRIEST': {
                'heal': '💊 Heal synergy - control value',
                'shadow': '👤 Shadow synergy - removal options',
                'divine': '✨ Divine synergy - board protection',
                'deathrattle': '💀 Deathrattle synergy - value engine'
            },
            'WARLOCK': {
                'demon': '👹 Demon synergy - aggressive stats',
                'discard': '🗑️ Discard synergy - risk/reward',
                'lifesteal': '🩸 Lifesteal synergy - health management',
                'self_damage': '⚡ Self-damage synergy - powerful effects'
            },
            'ROGUE': {
                'combo': '🎯 Combo synergy - efficient removal',
                'stealth': '👻 Stealth synergy - guaranteed value',
                'weapon': '🗡️ Weapon synergy - tempo swings',
                'deathrattle': '💀 Deathrattle synergy - value generation'
            },
            'SHAMAN': {
                'elemental': '🌊 Elemental synergy - chain effects',
                'overload': '⚡ Overload synergy - powerful early game',
                'weapon': '🔨 Weapon synergy - board control',
                'totem': '🗿 Totem synergy - board presence'
            },
            'PALADIN': {
                'divine_shield': '✨ Divine Shield synergy - sticky minions',
                'weapon': '⚔️ Weapon synergy - board control',
                'heal': '💊 Heal synergy - value trades',
                'buff': '💪 Buff synergy - board domination'
            },
            'DRUID': {
                'choose_one': '🌿 Choose One synergy - flexibility',
                'ramp': '🌱 Ramp synergy - powerful late game',
                'beast': '🐻 Beast synergy - tribal pressure',
                'armor': '🛡️ Armor synergy - survival tools'
            },
            'DEMONHUNTER': {
                'demon': '👹 Demon synergy - aggressive stats',
                'outcast': '🚀 Outcast synergy - powerful effects',
                'weapon': '⚔️ Weapon synergy - face damage',
                'attack': '💥 Attack synergy - hero power value'
            }
        }
        
        # Get synergies for this hero class
        hero_synergies = class_synergies.get(hero_class, {})
        
        # Try to identify card synergy type (simplified heuristic)
        card_lower = card_name.lower()
        for synergy_type, explanation in hero_synergies.items():
            if synergy_type in card_lower or any(keyword in card_lower for keyword in synergy_type.split('_')):
                return explanation
        
        # Generic class synergy
        return f"🎯 {hero_class.title()} class card - inherent synergy"
    
    def _get_neutral_synergy_explanation(self, card_code, card_name, hero_class):
        """Get neutral card synergy explanation for specific hero."""
        # Hero-specific neutral card preferences
        neutral_preferences = {
            'WARRIOR': {
                'taunt': '🛡️ Taunt - fits control gameplan',
                'weapon': '⚔️ Weapon - synergizes with hero power',
                'armor': '🛡️ Armor - defensive value',
                'high_health': '💪 High health - trading efficiency'
            },
            'MAGE': {
                'spell_damage': '✨ Spell damage - enhances removal',
                'freeze': '🧊 Freeze - tempo control',
                'draw': '📚 Card draw - resource advantage',
                'low_cost': '⚡ Low cost - efficient trades'
            },
            'HUNTER': {
                'charge': '🏃 Charge - immediate face damage',
                'beast': '🐺 Beast - tribal synergy',
                'low_cost': '⚡ Low cost - aggressive curve',
                'direct_damage': '🎯 Direct damage - reach potential'
            },
            'PRIEST': {
                'high_health': '💊 High health - heal synergy',
                'deathrattle': '💀 Deathrattle - value generation',
                'divine_shield': '✨ Divine Shield - protection',
                'card_draw': '📚 Card draw - control value'
            },
            'WARLOCK': {
                'demon': '👹 Demon - tribal synergy',
                'card_draw': '📚 Card draw - life tap value',
                'high_attack': '💥 High attack - aggressive pressure',
                'self_damage': '⚡ Self-damage - synergy potential'
            },
            'ROGUE': {
                'combo': '🎯 Combo - efficient removal',
                'low_cost': '⚡ Low cost - combo enabler',
                'stealth': '👻 Stealth - guaranteed value',
                'weapon': '🗡️ Weapon - tempo swings'
            },
            'SHAMAN': {
                'elemental': '🌊 Elemental - tribal chains',
                'spell_damage': '⚡ Spell damage - removal boost',
                'weapon': '🔨 Weapon - board control',
                'overload': '⚡ Overload - early power'
            },
            'PALADIN': {
                'divine_shield': '✨ Divine Shield - buff targets',
                'low_cost': '⚡ Low cost - flooding potential',
                'weapon': '⚔️ Weapon - board control',
                'heal': '💊 Heal - value trades'
            },
            'DRUID': {
                'high_cost': '🌱 High cost - ramp targets',
                'beast': '🐻 Beast - tribal value',
                'taunt': '🛡️ Taunt - defensive walls',
                'card_draw': '📚 Card draw - resource advantage'
            },
            'DEMONHUNTER': {
                'low_cost': '⚡ Low cost - aggressive curve',
                'demon': '👹 Demon - tribal synergy',
                'rush': '🏃 Rush - immediate impact',
                'weapon': '⚔️ Weapon - face damage'
            }
        }
        
        # Get preferences for this hero
        hero_prefs = neutral_preferences.get(hero_class, {})
        
        # Try to match card characteristics (simplified heuristic)
        card_lower = card_name.lower()
        for pref_type, explanation in hero_prefs.items():
            if pref_type in card_lower or any(keyword in card_lower for keyword in pref_type.split('_')):
                return explanation
        
        # Generic neutral assessment
        generic_assessments = {
            'WARRIOR': '⚖️ Neutral - control value potential',
            'MAGE': '⚖️ Neutral - tempo consideration',
            'HUNTER': '⚖️ Neutral - aggressive curve fit',
            'PRIEST': '⚖️ Neutral - value consideration',
            'WARLOCK': '⚖️ Neutral - aggressive potential',
            'ROGUE': '⚖️ Neutral - tempo efficiency',
            'SHAMAN': '⚖️ Neutral - midrange value',
            'PALADIN': '⚖️ Neutral - board presence',
            'DRUID': '⚖️ Neutral - flexible option',
            'DEMONHUNTER': '⚖️ Neutral - aggressive consideration'
        }
        
        return generic_assessments.get(hero_class, '⚖️ Neutral card')
    
    def setup_draft_review_panel(self):
        """Setup draft review panel showing hero choice impact on overall draft performance."""
        # Draft review frame (initially hidden)
        self.review_frame = tk.LabelFrame(
            self.root,
            text="📊 DRAFT REVIEW - HERO IMPACT ANALYSIS",
            font=('Arial', 10, 'bold'),
            fg='#9B59B6',
            bg='#2C3E50'
        )
        # Initially hidden - will show when draft is complete
        self.review_frame.pack_forget()
        
        # Create review container
        review_container = tk.Frame(self.review_frame, bg='#2C3E50')
        review_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Hero Impact Section
        hero_impact_section = tk.Frame(review_container, bg='#34495E', relief='raised', bd=2)
        hero_impact_section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        tk.Label(
            hero_impact_section,
            text="👑 HERO IMPACT",
            font=('Arial', 10, 'bold'),
            fg='#F39C12',
            bg='#34495E'
        ).pack(pady=5)
        
        self.hero_impact_text = tk.Text(
            hero_impact_section,
            height=8,
            width=25,
            bg='#2C3E50',
            fg='#ECF0F1',
            font=('Arial', 8),
            wrap=tk.WORD
        )
        self.hero_impact_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Draft Quality Section
        draft_quality_section = tk.Frame(review_container, bg='#34495E', relief='raised', bd=2)
        draft_quality_section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        tk.Label(
            draft_quality_section,
            text="📈 DRAFT QUALITY",
            font=('Arial', 10, 'bold'),
            fg='#3498DB',
            bg='#34495E'
        ).pack(pady=5)
        
        self.draft_quality_text = tk.Text(
            draft_quality_section,
            height=8,
            width=25,
            bg='#2C3E50',
            fg='#ECF0F1',
            font=('Arial', 8),
            wrap=tk.WORD
        )
        self.draft_quality_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Archetype Analysis Section
        archetype_section = tk.Frame(review_container, bg='#34495E', relief='raised', bd=2)
        archetype_section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        tk.Label(
            archetype_section,
            text="🎯 ARCHETYPE FIT",
            font=('Arial', 10, 'bold'),
            fg='#E74C3C',
            bg='#34495E'
        ).pack(pady=5)
        
        self.archetype_analysis_text = tk.Text(
            archetype_section,
            height=8,
            width=25,
            bg='#2C3E50',
            fg='#ECF0F1',
            font=('Arial', 8),
            wrap=tk.WORD
        )
        self.archetype_analysis_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Performance Prediction Section
        prediction_section = tk.Frame(review_container, bg='#34495E', relief='raised', bd=2)
        prediction_section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        tk.Label(
            prediction_section,
            text="🔮 PERFORMANCE PREDICTION",
            font=('Arial', 10, 'bold'),
            fg='#27AE60',
            bg='#34495E'
        ).pack(pady=5)
        
        self.performance_prediction_text = tk.Text(
            prediction_section,
            height=8,
            width=25,
            bg='#2C3E50',
            fg='#ECF0F1',
            font=('Arial', 8),
            wrap=tk.WORD
        )
        self.performance_prediction_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Review controls
        controls_frame = tk.Frame(self.review_frame, bg='#2C3E50')
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Export review button
        export_btn = tk.Button(
            controls_frame,
            text="📋 Export Review",
            command=self.export_draft_review,
            bg='#9B59B6',
            fg='white',
            font=('Arial', 9),
            relief='raised',
            bd=2
        )
        export_btn.pack(side='left', padx=5)
        
        # Hide review button
        hide_btn = tk.Button(
            controls_frame,
            text="❌ Hide Review",
            command=self.hide_draft_review,
            bg='#95A5A6',
            fg='white',
            font=('Arial', 9),
            relief='raised',
            bd=2
        )
        hide_btn.pack(side='right', padx=5)
    
    def show_draft_review(self):
        """Show and populate the draft review panel."""
        try:
            if not hasattr(self, 'review_frame'):
                return
            
            # Generate comprehensive review
            self._populate_hero_impact_analysis()
            self._populate_draft_quality_analysis()
            self._populate_archetype_analysis()
            self._populate_performance_prediction()
            
            # Show the review panel
            self.review_frame.pack(fill='both', expand=True, padx=10, pady=5)
            
            self.log_text("📊 Draft review panel displayed")
            
        except Exception as e:
            self.log_text(f"⚠️ Error showing draft review: {e}")
    
    def hide_draft_review(self):
        """Hide the draft review panel."""
        if hasattr(self, 'review_frame'):
            self.review_frame.pack_forget()
    
    def _populate_hero_impact_analysis(self):
        """Populate hero impact analysis."""
        try:
            self.hero_impact_text.delete('1.0', tk.END)
            
            if not self.selected_hero_class:
                self.hero_impact_text.insert('1.0', "No hero selected")
                return
            
            analysis = []
            analysis.append(f"Selected Hero: {self.selected_hero_class}\n")
            
            # Hero winrate impact
            if hasattr(self, 'hero_recommendation') and self.hero_recommendation:
                winrates = self.hero_recommendation.winrates
                if self.selected_hero_class in winrates:
                    hero_wr = winrates[self.selected_hero_class]
                    avg_wr = sum(winrates.values()) / len(winrates)
                    impact = hero_wr - avg_wr
                    
                    analysis.append(f"Hero Winrate: {hero_wr:.1f}%")
                    analysis.append(f"Average Offered: {avg_wr:.1f}%")
                    analysis.append(f"Impact: {impact:+.1f}%\n")
                    
                    if impact > 2:
                        analysis.append("🎯 Excellent hero choice!")
                    elif impact > 0:
                        analysis.append("✅ Good hero choice")
                    elif impact > -2:
                        analysis.append("⚖️ Average hero choice")
                    else:
                        analysis.append("⚠️ Suboptimal hero choice")
            
            # Archetype compatibility
            if hasattr(self, 'grandmaster_advisor') and self.grandmaster_advisor:
                hero_affinities = self.grandmaster_advisor._calculate_hero_archetype_preferences(self.selected_hero_class)
                analysis.append(f"\nArchetype Affinities:")
                for archetype, affinity in sorted(hero_affinities.items(), key=lambda x: x[1], reverse=True)[:3]:
                    analysis.append(f"• {archetype}: {affinity:.1%}")
            
            # Card synergy potential
            analysis.append(f"\nSynergy Potential:")
            analysis.append(f"• Class cards: High synergy")
            analysis.append(f"• Neutral cards: Hero-specific value")
            analysis.append(f"• Total picks: {getattr(self, 'draft_picks_count', 0)}/30")
            
            self.hero_impact_text.insert('1.0', '\n'.join(analysis))
            
        except Exception as e:
            self.hero_impact_text.insert('1.0', f"Error analyzing hero impact: {e}")
    
    def _populate_draft_quality_analysis(self):
        """Populate draft quality analysis."""
        try:
            self.draft_quality_text.delete('1.0', tk.END)
            
            analysis = []
            pick_count = getattr(self, 'draft_picks_count', 0)
            
            analysis.append(f"Draft Progress: {pick_count}/30\n")
            
            # AI v2 system performance
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                try:
                    performance = self.ai_v2_integrator._get_performance_summary()
                    card_evals = performance.get('card_evaluations_last_hour', 0)
                    hero_recs = performance.get('hero_recommendations_last_hour', 0)
                    
                    analysis.append("AI v2 Performance:")
                    analysis.append(f"• Hero recs: {hero_recs}")
                    analysis.append(f"• Card evals: {card_evals}")
                    
                    avg_eval_time = performance.get('avg_card_eval_time_ms', 0)
                    if avg_eval_time > 0:
                        analysis.append(f"• Avg eval time: {avg_eval_time:.0f}ms")
                except:
                    analysis.append("AI v2 Performance: Unknown")
            
            # Data source quality
            analysis.append(f"\nData Source Quality:")
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                try:
                    health = self.ai_v2_integrator.check_system_health()
                    overall_status = health.get('overall_status', 'unknown')
                    if overall_status == 'online':
                        analysis.append("• HSReplay: ✅ Online")
                        analysis.append("• Full AI v2: ✅ Active")
                    else:
                        analysis.append("• HSReplay: ⚠️ Limited")
                        analysis.append("• Fallback: 💾 Active")
                except:
                    analysis.append("• Status: ❓ Unknown")
            
            # Draft efficiency estimate
            if pick_count > 5:
                efficiency = min(95, 70 + (pick_count * 0.5))  # Simplified calculation
                analysis.append(f"\nDraft Efficiency: {efficiency:.0f}%")
                
                if efficiency > 85:
                    analysis.append("🎯 Excellent draft quality")
                elif efficiency > 75:
                    analysis.append("✅ Good draft quality")
                elif efficiency > 65:
                    analysis.append("⚖️ Average draft quality")
                else:
                    analysis.append("⚠️ Room for improvement")
            
            self.draft_quality_text.insert('1.0', '\n'.join(analysis))
            
        except Exception as e:
            self.draft_quality_text.insert('1.0', f"Error analyzing draft quality: {e}")
    
    def _populate_archetype_analysis(self):
        """Populate archetype fit analysis."""
        try:
            self.archetype_analysis_text.delete('1.0', tk.END)
            
            analysis = []
            
            if not self.selected_hero_class:
                analysis.append("No hero selected for analysis")
            else:
                # Current archetype (default to Balanced)
                current_archetype = getattr(self, 'current_archetype', 'Balanced')
                analysis.append(f"Current: {current_archetype}\n")
                
                # Hero affinity for current archetype
                if hasattr(self, 'grandmaster_advisor') and self.grandmaster_advisor:
                    hero_affinities = self.grandmaster_advisor._calculate_hero_archetype_preferences(self.selected_hero_class)
                    current_affinity = hero_affinities.get(current_archetype, 0.5)
                    
                    analysis.append(f"Hero Affinity: {current_affinity:.1%}")
                    
                    if current_affinity > 0.8:
                        analysis.append("🎯 Perfect archetype match!")
                    elif current_affinity > 0.6:
                        analysis.append("✅ Good archetype fit")
                    elif current_affinity > 0.4:
                        analysis.append("⚖️ Acceptable fit")
                    else:
                        analysis.append("⚠️ Poor archetype fit")
                    
                    # Best alternative archetypes
                    analysis.append(f"\nBest Alternatives:")
                    sorted_archetypes = sorted(hero_affinities.items(), key=lambda x: x[1], reverse=True)
                    for archetype, affinity in sorted_archetypes[:3]:
                        if archetype != current_archetype:
                            analysis.append(f"• {archetype}: {affinity:.1%}")
                
                # Archetype consistency based on picks
                pick_count = getattr(self, 'draft_picks_count', 0)
                if pick_count > 10:
                    # Simplified consistency calculation
                    consistency = min(90, 60 + (pick_count * 1.5))
                    analysis.append(f"\nConsistency: {consistency:.0f}%")
                    
                    if consistency > 80:
                        analysis.append("🎯 Highly consistent")
                    elif consistency > 70:
                        analysis.append("✅ Good consistency")
                    else:
                        analysis.append("⚠️ Mixed signals")
                
                # Pivot recommendations
                analysis.append(f"\nPivot Analysis:")
                if pick_count < 15:
                    analysis.append("• Still flexible")
                    analysis.append("• Can pivot if needed")
                elif pick_count < 25:
                    analysis.append("• Limited pivot options")
                    analysis.append("• Stay committed")
                else:
                    analysis.append("• No pivot possible")
                    analysis.append("• Finalize strategy")
            
            self.archetype_analysis_text.insert('1.0', '\n'.join(analysis))
            
        except Exception as e:
            self.archetype_analysis_text.insert('1.0', f"Error analyzing archetype: {e}")
    
    def _populate_performance_prediction(self):
        """Populate performance prediction analysis."""
        try:
            self.performance_prediction_text.delete('1.0', tk.END)
            
            analysis = []
            
            if not self.selected_hero_class:
                analysis.append("No data for prediction")
                self.performance_prediction_text.insert('1.0', '\n'.join(analysis))
                return
            
            # Base prediction from hero winrate
            base_prediction = 50.0  # Default
            if hasattr(self, 'hero_recommendation') and self.hero_recommendation:
                winrates = self.hero_recommendation.winrates
                if self.selected_hero_class in winrates:
                    base_prediction = winrates[self.selected_hero_class]
            
            analysis.append(f"Base Hero WR: {base_prediction:.1f}%")
            
            # Adjustments based on draft quality
            pick_count = getattr(self, 'draft_picks_count', 0)
            if pick_count > 0:
                # Simplified draft quality modifier
                draft_modifier = 0
                
                # AI v2 usage bonus
                if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                    try:
                        health = self.ai_v2_integrator.check_system_health()
                        if health.get('overall_status') == 'online':
                            draft_modifier += 2  # AI v2 bonus
                        else:
                            draft_modifier += 1  # Partial bonus
                    except:
                        pass
                
                # Pick count modifier (more picks = better draft)
                pick_modifier = min(3, pick_count * 0.1)
                draft_modifier += pick_modifier
                
                adjusted_prediction = base_prediction + draft_modifier
                
                analysis.append(f"Draft Bonus: +{draft_modifier:.1f}%")
                analysis.append(f"Adjusted WR: {adjusted_prediction:.1f}%\n")
                
                # Win prediction ranges
                analysis.append("Expected Wins:")
                low_wins = max(0, int((adjusted_prediction - 5) / 10))
                high_wins = min(12, int((adjusted_prediction + 5) / 10))
                most_likely = int(adjusted_prediction / 10)
                
                analysis.append(f"• Most likely: {most_likely} wins")
                analysis.append(f"• Range: {low_wins}-{high_wins} wins")
                
                # Performance categories
                if most_likely >= 7:
                    analysis.append(f"• Category: 🏆 Excellent")
                elif most_likely >= 5:
                    analysis.append(f"• Category: ✅ Good")
                elif most_likely >= 3:
                    analysis.append(f"• Category: ⚖️ Average")
                else:
                    analysis.append(f"• Category: ⚠️ Challenging")
                
                # Confidence in prediction
                confidence = 70 + min(20, pick_count)  # More picks = higher confidence
                analysis.append(f"\nPrediction Confidence: {confidence}%")
                
                # Factors affecting performance
                analysis.append(f"\nKey Factors:")
                analysis.append(f"• Hero choice impact")
                analysis.append(f"• AI-guided picks")
                analysis.append(f"• Archetype consistency")
                analysis.append(f"• Player skill")
            
            self.performance_prediction_text.insert('1.0', '\n'.join(analysis))
            
        except Exception as e:
            self.performance_prediction_text.insert('1.0', f"Error predicting performance: {e}")
    
    def export_draft_review(self):
        """Export the complete draft review to a file."""
        try:
            from datetime import datetime
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hero_name = self.selected_hero_class or "Unknown"
            filename = f"draft_review_{hero_name}_{timestamp}.txt"
            
            # Collect all review data
            review_data = []
            review_data.append("=" * 60)
            review_data.append("ARENA BOT AI v2 - DRAFT REVIEW")
            review_data.append("=" * 60)
            review_data.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            review_data.append(f"Hero: {hero_name}")
            review_data.append(f"Picks: {getattr(self, 'draft_picks_count', 0)}/30")
            review_data.append("")
            
            # Hero Impact
            review_data.append("HERO IMPACT ANALYSIS:")
            review_data.append("-" * 30)
            if hasattr(self, 'hero_impact_text'):
                hero_content = self.hero_impact_text.get('1.0', tk.END).strip()
                review_data.append(hero_content)
            review_data.append("")
            
            # Draft Quality
            review_data.append("DRAFT QUALITY ANALYSIS:")
            review_data.append("-" * 30)
            if hasattr(self, 'draft_quality_text'):
                quality_content = self.draft_quality_text.get('1.0', tk.END).strip()
                review_data.append(quality_content)
            review_data.append("")
            
            # Archetype Analysis
            review_data.append("ARCHETYPE FIT ANALYSIS:")
            review_data.append("-" * 30)
            if hasattr(self, 'archetype_analysis_text'):
                archetype_content = self.archetype_analysis_text.get('1.0', tk.END).strip()
                review_data.append(archetype_content)
            review_data.append("")
            
            # Performance Prediction
            review_data.append("PERFORMANCE PREDICTION:")
            review_data.append("-" * 30)
            if hasattr(self, 'performance_prediction_text'):
                prediction_content = self.performance_prediction_text.get('1.0', tk.END).strip()
                review_data.append(prediction_content)
            review_data.append("")
            
            review_data.append("=" * 60)
            review_data.append("Generated by Arena Bot AI v2 System")
            review_data.append("https://github.com/arena-bot-project")
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(review_data))
            
            self.log_text(f"📋 Draft review exported to: {filename}")
            
        except Exception as e:
            self.log_text(f"❌ Error exporting review: {e}")
    
    def show_unified_statistics(self):
        """Show unified statistics window covering both hero and card performance data."""
        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title("📈 AI v2 Unified Statistics")
            stats_window.geometry("1200x800")
            stats_window.configure(bg='#2C3E50')
            
            # Make stats window stay on top
            stats_window.attributes('-topmost', True)
            
            # Main title
            title_label = tk.Label(
                stats_window,
                text="📈 AI v2 UNIFIED STATISTICS DASHBOARD",
                font=('Arial', 16, 'bold'),
                fg='#E74C3C',
                bg='#2C3E50'
            )
            title_label.pack(pady=10)
            
            # Create main container with tabs
            main_container = tk.Frame(stats_window, bg='#2C3E50')
            main_container.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Create notebook for tabs
            from tkinter import ttk
            style = ttk.Style()
            style.theme_use('clam')
            
            notebook = ttk.Notebook(main_container)
            notebook.pack(fill='both', expand=True)
            
            # System Overview Tab
            overview_frame = tk.Frame(notebook, bg='#34495E')
            notebook.add(overview_frame, text="🎯 System Overview")
            self._create_system_overview_tab(overview_frame)
            
            # Hero Performance Tab
            hero_frame = tk.Frame(notebook, bg='#34495E')
            notebook.add(hero_frame, text="👑 Hero Performance")
            self._create_hero_performance_tab(hero_frame)
            
            # Card Analysis Tab
            card_frame = tk.Frame(notebook, bg='#34495E')
            notebook.add(card_frame, text="🃏 Card Analysis")
            self._create_card_analysis_tab(card_frame)
            
            # Data Sources Tab
            data_frame = tk.Frame(notebook, bg='#34495E')
            notebook.add(data_frame, text="🔧 Data Sources")
            self._create_data_sources_tab(data_frame)
            
            # Performance Metrics Tab
            perf_frame = tk.Frame(notebook, bg='#34495E')
            notebook.add(perf_frame, text="⚡ Performance")
            self._create_performance_metrics_tab(perf_frame)
            
            # Control buttons
            controls_frame = tk.Frame(stats_window, bg='#2C3E50')
            controls_frame.pack(fill='x', padx=10, pady=5)
            
            # Refresh button
            refresh_btn = tk.Button(
                controls_frame,
                text="🔄 Refresh",
                command=lambda: self._refresh_statistics_display(stats_window),
                bg='#3498DB',
                fg='white',
                font=('Arial', 10),
                relief='raised',
                bd=2
            )
            refresh_btn.pack(side='left', padx=5)
            
            # Export button
            export_btn = tk.Button(
                controls_frame,
                text="📋 Export Stats",
                command=self.export_unified_statistics,
                bg='#27AE60',
                fg='white',
                font=('Arial', 10),
                relief='raised',
                bd=2
            )
            export_btn.pack(side='left', padx=5)
            
            # Close button
            close_btn = tk.Button(
                controls_frame,
                text="❌ Close",
                command=stats_window.destroy,
                bg='#E74C3C',
                fg='white',
                font=('Arial', 10),
                relief='raised',
                bd=2
            )
            close_btn.pack(side='right', padx=5)
            
            self.log_text("📈 AI v2 Unified Statistics displayed")
            
        except Exception as e:
            self.log_text(f"❌ Error showing unified statistics: {e}")
    
    def _create_system_overview_tab(self, parent):
        """Create system overview tab with key metrics."""
        # Overall system status
        status_frame = tk.LabelFrame(parent, text="🎯 System Status", bg='#34495E', fg='#ECF0F1', font=('Arial', 10, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=5)
        
        status_text = tk.Text(status_frame, height=8, bg='#2C3E50', fg='#ECF0F1', font=('Arial', 9))
        status_text.pack(fill='x', padx=5, pady=5)
        
        # Get system status
        try:
            status_info = []
            status_info.append("AI v2 SYSTEM STATUS")
            status_info.append("=" * 30)
            
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                health = self.ai_v2_integrator.check_system_health()
                overall_status = health.get('overall_status', 'unknown')
                status_info.append(f"Overall Status: {overall_status.upper()}")
                
                components = health.get('components', {})
                status_info.append(f"\nComponent Status:")
                for comp, details in components.items():
                    if isinstance(details, dict):
                        comp_status = details.get('status', 'unknown')
                        response_time = details.get('response_time_ms', 0)
                        status_info.append(f"• {comp}: {comp_status} ({response_time:.0f}ms)")
                
                uptime = health.get('uptime_hours', 0)
                status_info.append(f"\nSystem Uptime: {uptime:.1f} hours")
            else:
                status_info.append("AI v2 System: OFFLINE")
            
            # Current draft info
            status_info.append(f"\nCurrent Session:")
            status_info.append(f"• Hero: {getattr(self, 'selected_hero_class', 'None')}")
            status_info.append(f"• Phase: {getattr(self, 'current_draft_phase', 'waiting')}")
            status_info.append(f"• Picks: {getattr(self, 'draft_picks_count', 0)}/30")
            
            status_text.insert('1.0', '\n'.join(status_info))
            
        except Exception as e:
            status_text.insert('1.0', f"Error loading system status: {e}")
        
        # Quick stats
        quick_stats_frame = tk.LabelFrame(parent, text="📊 Quick Statistics", bg='#34495E', fg='#ECF0F1', font=('Arial', 10, 'bold'))
        quick_stats_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create grid for quick stats
        stats_container = tk.Frame(quick_stats_frame, bg='#34495E')
        stats_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Hero stats
        hero_stats_frame = tk.Frame(stats_container, bg='#2C3E50', relief='raised', bd=2)
        hero_stats_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        tk.Label(hero_stats_frame, text="👑 HERO STATS", bg='#2C3E50', fg='#F39C12', font=('Arial', 10, 'bold')).pack(pady=5)
        
        hero_stats_text = tk.Text(hero_stats_frame, height=10, bg='#2C3E50', fg='#ECF0F1', font=('Arial', 8))
        hero_stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        try:
            hero_info = []
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                performance = self.ai_v2_integrator._get_performance_summary()
                hero_recs = performance.get('hero_recommendations_last_hour', 0)
                hero_info.append(f"Recommendations: {hero_recs}")
                
                avg_time = performance.get('avg_hero_response_time_ms', 0)
                hero_info.append(f"Avg Response: {avg_time:.0f}ms")
            
            if hasattr(self, 'hero_recommendation') and self.hero_recommendation:
                confidence = self.hero_recommendation.confidence_level
                hero_info.append(f"Last Confidence: {confidence:.1%}")
                
                winrates = self.hero_recommendation.winrates
                if winrates:
                    avg_wr = sum(winrates.values()) / len(winrates)
                    hero_info.append(f"Avg Winrate: {avg_wr:.1f}%")
            
            if not hero_info:
                hero_info.append("No hero data available")
            
            hero_stats_text.insert('1.0', '\n'.join(hero_info))
            
        except Exception as e:
            hero_stats_text.insert('1.0', f"Error: {e}")
        
        # Card stats
        card_stats_frame = tk.Frame(stats_container, bg='#2C3E50', relief='raised', bd=2)
        card_stats_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        tk.Label(card_stats_frame, text="🃏 CARD STATS", bg='#2C3E50', fg='#3498DB', font=('Arial', 10, 'bold')).pack(pady=5)
        
        card_stats_text = tk.Text(card_stats_frame, height=10, bg='#2C3E50', fg='#ECF0F1', font=('Arial', 8))
        card_stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        try:
            card_info = []
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                performance = self.ai_v2_integrator._get_performance_summary()
                card_evals = performance.get('card_evaluations_last_hour', 0)
                card_info.append(f"Evaluations: {card_evals}")
                
                avg_time = performance.get('avg_card_eval_time_ms', 0)
                card_info.append(f"Avg Eval Time: {avg_time:.0f}ms")
            
            pick_count = getattr(self, 'draft_picks_count', 0)
            card_info.append(f"Current Picks: {pick_count}")
            
            if not card_info or all(line.endswith(': 0') for line in card_info):
                card_info = ["No card data available"]
            
            card_stats_text.insert('1.0', '\n'.join(card_info))
            
        except Exception as e:
            card_stats_text.insert('1.0', f"Error: {e}")
    
    def _create_hero_performance_tab(self, parent):
        """Create hero performance analysis tab."""
        # Hero comparison frame
        comparison_frame = tk.LabelFrame(parent, text="👑 Hero Winrate Comparison", bg='#34495E', fg='#ECF0F1', font=('Arial', 10, 'bold'))
        comparison_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        comparison_text = tk.Text(comparison_frame, bg='#2C3E50', fg='#ECF0F1', font=('Consolas', 9))
        comparison_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        try:
            hero_data = []
            hero_data.append("HERO WINRATE ANALYSIS")
            hero_data.append("=" * 50)
            
            if hasattr(self, 'hero_recommendation') and self.hero_recommendation:
                winrates = self.hero_recommendation.winrates
                
                if winrates:
                    # Sort heroes by winrate
                    sorted_heroes = sorted(winrates.items(), key=lambda x: x[1], reverse=True)
                    
                    hero_data.append(f"{'Hero':<15} {'Winrate':<10} {'Tier':<10} {'Status'}")
                    hero_data.append("-" * 50)
                    
                    for hero, winrate in sorted_heroes:
                        # Determine tier
                        if winrate >= 55:
                            tier = "S-Tier"
                            color = "🟢"
                        elif winrate >= 52:
                            tier = "A-Tier"
                            color = "🟡"
                        elif winrate >= 48:
                            tier = "B-Tier"
                            color = "🟠"
                        else:
                            tier = "C-Tier"
                            color = "🔴"
                        
                        # Check if this is the selected hero
                        status = "Selected" if hero == getattr(self, 'selected_hero_class', None) else ""
                        
                        hero_data.append(f"{hero:<15} {winrate:>6.1f}%   {color} {tier:<8} {status}")
                    
                    # Add statistics
                    avg_winrate = sum(winrates.values()) / len(winrates)
                    hero_data.append("")
                    hero_data.append(f"Average Winrate: {avg_winrate:.1f}%")
                    hero_data.append(f"Best Hero: {sorted_heroes[0][0]} ({sorted_heroes[0][1]:.1f}%)")
                    hero_data.append(f"Worst Hero: {sorted_heroes[-1][0]} ({sorted_heroes[-1][1]:.1f}%)")
                    hero_data.append(f"Winrate Spread: {sorted_heroes[0][1] - sorted_heroes[-1][1]:.1f}%")
                else:
                    hero_data.append("No hero winrate data available")
            else:
                hero_data.append("No hero recommendation data available")
            
            comparison_text.insert('1.0', '\n'.join(hero_data))
            
        except Exception as e:
            comparison_text.insert('1.0', f"Error loading hero data: {e}")
    
    def _create_card_analysis_tab(self, parent):
        """Create card analysis tab."""
        # Card evaluation metrics
        eval_frame = tk.LabelFrame(parent, text="🃏 Card Evaluation Metrics", bg='#34495E', fg='#ECF0F1', font=('Arial', 10, 'bold'))
        eval_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        eval_text = tk.Text(eval_frame, bg='#2C3E50', fg='#ECF0F1', font=('Consolas', 9))
        eval_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        try:
            card_data = []
            card_data.append("CARD EVALUATION METRICS")
            card_data.append("=" * 50)
            
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                # Get advisor statistics
                if hasattr(self, 'grandmaster_advisor') and self.grandmaster_advisor:
                    stats = self.grandmaster_advisor.get_advisor_statistics()
                    
                    card_data.append(f"Recommendations Made: {stats.get('recommendations_made', 0)}")
                    card_data.append(f"Current Hero Context: {stats.get('current_hero', 'None')}")
                    card_data.append(f"Pivot Opportunities Found: {stats.get('pivot_opportunities_found', 0)}")
                    card_data.append(f"Average Confidence: {stats.get('avg_confidence', 0):.1%}")
                    card_data.append(f"Last Analysis Time: {stats.get('last_analysis_time_ms', 0):.0f}ms")
                    
                    card_data.append("\nArchetype Weights (Current Hero):")
                    hero_weights = stats.get('hero_archetype_weights', {})
                    if hero_weights:
                        for archetype, weight in sorted(hero_weights.items(), key=lambda x: x[1], reverse=True):
                            card_data.append(f"• {archetype}: {weight:.1%}")
                    
                    card_data.append("\nSystem Integration:")
                    integration = stats.get('system_integration', {})
                    for system, available in integration.items():
                        status = "✅ Online" if available else "❌ Offline"
                        card_data.append(f"• {system}: {status}")
                
                # Performance metrics
                performance = self.ai_v2_integrator._get_performance_summary()
                card_data.append(f"\nPerformance (Last Hour):")
                card_data.append(f"• Card Evaluations: {performance.get('card_evaluations_last_hour', 0)}")
                card_data.append(f"• Avg Evaluation Time: {performance.get('avg_card_eval_time_ms', 0):.0f}ms")
                card_data.append(f"• Total Errors: {performance.get('total_errors_last_hour', 0)}")
            else:
                card_data.append("AI v2 system not available")
            
            eval_text.insert('1.0', '\n'.join(card_data))
            
        except Exception as e:
            eval_text.insert('1.0', f"Error loading card data: {e}")
    
    def _create_data_sources_tab(self, parent):
        """Create data sources status tab."""
        # Data source status
        sources_frame = tk.LabelFrame(parent, text="🔧 Data Source Status", bg='#34495E', fg='#ECF0F1', font=('Arial', 10, 'bold'))
        sources_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        sources_text = tk.Text(sources_frame, bg='#2C3E50', fg='#ECF0F1', font=('Consolas', 9))
        sources_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        try:
            source_data = []
            source_data.append("DATA SOURCE STATUS REPORT")
            source_data.append("=" * 60)
            
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                health = self.ai_v2_integrator.check_system_health()
                data_sources = health.get('data_sources', {})
                
                # HSReplay status
                if 'hsreplay' in data_sources:
                    hsreplay = data_sources['hsreplay']
                    source_data.append("HSReplay API:")
                    source_data.append(f"• Status: {hsreplay.get('status', 'unknown').upper()}")
                    source_data.append(f"• API Calls Made: {hsreplay.get('api_calls', 0)}")
                    source_data.append(f"• Card Cache Age: {hsreplay.get('card_cache_age_hours', 0):.1f} hours")
                    source_data.append(f"• Hero Cache Age: {hsreplay.get('hero_cache_age_hours', 0):.1f} hours")
                
                # Cards database status
                if 'cards_database' in data_sources:
                    cards_db = data_sources['cards_database']
                    source_data.append("\nCards Database:")
                    source_data.append(f"• Status: {cards_db.get('status', 'unknown').upper()}")
                    source_data.append(f"• Total Cards: {cards_db.get('total_cards', 0)}")
                    source_data.append(f"• DBF Mappings: {cards_db.get('dbf_mappings', 0)}")
                
                # Component health
                source_data.append("\nComponent Health:")
                components = health.get('components', {})
                for comp_name, comp_details in components.items():
                    if isinstance(comp_details, dict):
                        status = comp_details.get('status', 'unknown')
                        error_count = comp_details.get('error_count', 0)
                        fallback = comp_details.get('fallback_active', False)
                        
                        source_data.append(f"• {comp_name}: {status.upper()}")
                        if error_count > 0:
                            source_data.append(f"  └─ Errors: {error_count}")
                        if fallback:
                            source_data.append(f"  └─ Fallback: ACTIVE")
                
                # Overall performance
                perf = health.get('performance', {})
                if perf:
                    source_data.append(f"\nSystem Performance:")
                    for metric, value in perf.items():
                        source_data.append(f"• {metric}: {value}")
            else:
                source_data.append("AI v2 system not available")
                source_data.append("Cannot retrieve data source status")
            
            sources_text.insert('1.0', '\n'.join(source_data))
            
        except Exception as e:
            sources_text.insert('1.0', f"Error loading data source status: {e}")
    
    def _create_performance_metrics_tab(self, parent):
        """Create performance metrics tab."""
        # Performance overview
        perf_frame = tk.LabelFrame(parent, text="⚡ Performance Overview", bg='#34495E', fg='#ECF0F1', font=('Arial', 10, 'bold'))
        perf_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        perf_text = tk.Text(perf_frame, bg='#2C3E50', fg='#ECF0F1', font=('Consolas', 9))
        perf_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        try:
            perf_data = []
            perf_data.append("PERFORMANCE METRICS DASHBOARD")
            perf_data.append("=" * 60)
            
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                performance = self.ai_v2_integrator._get_performance_summary()
                
                perf_data.append("Response Times (Last Hour):")
                hero_time = performance.get('avg_hero_response_time_ms', 0)
                card_time = performance.get('avg_card_eval_time_ms', 0)
                
                perf_data.append(f"• Hero Recommendations: {hero_time:.0f}ms")
                perf_data.append(f"• Card Evaluations: {card_time:.0f}ms")
                
                # Performance ratings
                def get_performance_rating(time_ms):
                    if time_ms < 100:
                        return "🟢 Excellent"
                    elif time_ms < 500:
                        return "🟡 Good"
                    elif time_ms < 1000:
                        return "🟠 Acceptable"
                    else:
                        return "🔴 Slow"
                
                perf_data.append(f"• Hero Performance: {get_performance_rating(hero_time)}")
                perf_data.append(f"• Card Performance: {get_performance_rating(card_time)}")
                
                # Activity metrics
                perf_data.append("\nActivity (Last Hour):")
                hero_recs = performance.get('hero_recommendations_last_hour', 0)
                card_evals = performance.get('card_evaluations_last_hour', 0)
                errors = performance.get('total_errors_last_hour', 0)
                
                perf_data.append(f"• Hero Recommendations: {hero_recs}")
                perf_data.append(f"• Card Evaluations: {card_evals}")
                perf_data.append(f"• Total Operations: {hero_recs + card_evals}")
                perf_data.append(f"• Errors: {errors}")
                
                # Error rate
                total_ops = hero_recs + card_evals
                if total_ops > 0:
                    error_rate = (errors / total_ops) * 100
                    perf_data.append(f"• Error Rate: {error_rate:.1f}%")
                    
                    if error_rate < 1:
                        perf_data.append("• Reliability: 🟢 Excellent")
                    elif error_rate < 5:
                        perf_data.append("• Reliability: 🟡 Good")
                    elif error_rate < 10:
                        perf_data.append("• Reliability: 🟠 Fair")
                    else:
                        perf_data.append("• Reliability: 🔴 Poor")
                
                # System health score
                health_score = 100
                if hero_time > 500:
                    health_score -= 20
                if card_time > 500:
                    health_score -= 20
                if total_ops > 0 and (errors / total_ops) > 0.05:
                    health_score -= 30
                
                perf_data.append(f"\nSystem Health Score: {health_score}/100")
                
                if health_score >= 90:
                    perf_data.append("Overall Rating: 🟢 Excellent")
                elif health_score >= 75:
                    perf_data.append("Overall Rating: 🟡 Good")
                elif health_score >= 60:
                    perf_data.append("Overall Rating: 🟠 Fair")
                else:
                    perf_data.append("Overall Rating: 🔴 Poor")
            else:
                perf_data.append("AI v2 system not available")
                perf_data.append("Cannot retrieve performance metrics")
            
            perf_text.insert('1.0', '\n'.join(perf_data))
            
        except Exception as e:
            perf_text.insert('1.0', f"Error loading performance data: {e}")
    
    def _refresh_statistics_display(self, stats_window):
        """Refresh the statistics display."""
        try:
            # Close and reopen the window
            stats_window.destroy()
            self.show_unified_statistics()
            self.log_text("🔄 Statistics refreshed")
        except Exception as e:
            self.log_text(f"❌ Error refreshing statistics: {e}")
    
    def export_unified_statistics(self):
        """Export unified statistics to a file."""
        try:
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_v2_statistics_{timestamp}.txt"
            
            stats_data = []
            stats_data.append("=" * 80)
            stats_data.append("ARENA BOT AI v2 - UNIFIED STATISTICS REPORT")
            stats_data.append("=" * 80)
            stats_data.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            stats_data.append("")
            
            # System status
            stats_data.append("SYSTEM STATUS:")
            stats_data.append("-" * 40)
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                health = self.ai_v2_integrator.check_system_health()
                stats_data.append(f"Overall Status: {health.get('overall_status', 'unknown').upper()}")
                stats_data.append(f"Uptime: {health.get('uptime_hours', 0):.1f} hours")
            stats_data.append("")
            
            # Performance summary
            stats_data.append("PERFORMANCE SUMMARY:")
            stats_data.append("-" * 40)
            if hasattr(self, 'ai_v2_integrator') and self.ai_v2_integrator:
                performance = self.ai_v2_integrator._get_performance_summary()
                hero_recs = performance.get('hero_recommendations_last_hour', 0)
                card_evals = performance.get('card_evaluations_last_hour', 0)
                hero_time = performance.get('avg_hero_response_time_ms', 0)
                card_time = performance.get('avg_card_eval_time_ms', 0)
                
                stats_data.append(f"Hero Recommendations (1h): {hero_recs}")
                stats_data.append(f"Card Evaluations (1h): {card_evals}")
                stats_data.append(f"Avg Hero Response Time: {hero_time:.0f}ms")
                stats_data.append(f"Avg Card Eval Time: {card_time:.0f}ms")
            stats_data.append("")
            
            # Current session
            stats_data.append("CURRENT SESSION:")
            stats_data.append("-" * 40)
            stats_data.append(f"Hero: {getattr(self, 'selected_hero_class', 'None')}")
            stats_data.append(f"Phase: {getattr(self, 'current_draft_phase', 'waiting')}")
            stats_data.append(f"Picks: {getattr(self, 'draft_picks_count', 0)}/30")
            stats_data.append("")
            
            # Hero data
            if hasattr(self, 'hero_recommendation') and self.hero_recommendation:
                stats_data.append("HERO WINRATES:")
                stats_data.append("-" * 40)
                winrates = self.hero_recommendation.winrates
                for hero, winrate in sorted(winrates.items(), key=lambda x: x[1], reverse=True):
                    stats_data.append(f"{hero}: {winrate:.1f}%")
            
            stats_data.append("")
            stats_data.append("=" * 80)
            stats_data.append("Generated by Arena Bot AI v2 System")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(stats_data))
            
            self.log_text(f"📋 Statistics exported to: {filename}")
            
        except Exception as e:
            self.log_text(f"❌ Error exporting statistics: {e}")
    
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
    
    # === NEW AI v2 INTEGRATION METHODS ===
    
    def _handle_hero_selection(self, hero_classes: List[str]):
        """Handle hero selection with AI v2 system."""
        try:
            if not self.hero_selector:
                self.log_text("⚠️ Hero selection AI not available")
                return
            
            if not hero_classes:
                self.log_text("👑 Hero classes not detected, using fallback")
                hero_classes = ["WARRIOR", "MAGE", "PALADIN"]  # Default fallback
            
            # Get AI v2 hero recommendation
            self.log_text("🎯 Analyzing hero options with AI v2...")
            hero_recommendation = self.ai_v2_integrator.get_hero_recommendation_with_recovery(hero_classes)
            
            # Display hero selection UI
            self._display_hero_selection_ui(hero_recommendation)
            
            # Log recommendation details
            recommended_hero = hero_recommendation.hero_classes[hero_recommendation.recommended_hero_index]
            confidence = hero_recommendation.confidence_level
            
            self.log_text(f"🎯 AI v2 HERO RECOMMENDATION:")
            self.log_text(f"   Recommended: {recommended_hero} ({confidence:.1%} confidence)")
            self.log_text(f"   Explanation: {hero_recommendation.explanation}")
            
        except Exception as e:
            self.log_text(f"❌ Hero selection error: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_enhanced_hero_selection(self, hero_classes: List[str], hero_data: dict):
        """Enhanced hero selection with full context and phase management."""
        try:
            if not self.ai_v2_integrator:
                self.log_text("⚠️ AI v2 system not available")
                return
            
            # Store the raw hero data for additional context
            self.current_hero_data = hero_data
            
            if not hero_classes:
                self.log_text("👑 Hero classes not detected, using fallback")
                hero_classes = ["WARRIOR", "MAGE", "PALADIN"]  # Default fallback
            
            # Get AI v2 hero recommendation with enhanced error recovery
            self.log_text("🎯 Analyzing hero options with AI v2 (enhanced)...")
            hero_recommendation = self.ai_v2_integrator.get_hero_recommendation_with_recovery(hero_classes)
            
            # Store recommendation for later use
            self.hero_recommendation = hero_recommendation
            
            # Display enhanced hero selection UI
            self._display_hero_selection_ui(hero_recommendation)
            
            # Log detailed recommendation
            recommended_hero = hero_recommendation.hero_classes[hero_recommendation.recommended_hero_index]
            confidence = hero_recommendation.confidence_level
            
            self.log_text(f"🎯 AI v2 ENHANCED HERO RECOMMENDATION:")
            self.log_text(f"   Recommended: {recommended_hero} ({confidence:.1%} confidence)")
            self.log_text(f"   Explanation: {hero_recommendation.explanation}")
            
            # Show system health for transparency
            self._show_system_health()
            
            # Prepare for transition to card picks
            self._prepare_for_card_draft_phase(recommended_hero)
            
        except Exception as e:
            self.log_text(f"❌ Enhanced hero selection error: {e}")
            import traceback
            traceback.print_exc()
    
    def _prepare_for_card_draft_phase(self, recommended_hero: str):
        """Prepare the system for transitioning to card draft phase."""
        try:
            # Store recommended hero for context
            self.selected_hero_class = recommended_hero
            
            # Initialize hero-aware card evaluation context
            if hasattr(self, 'grandmaster_advisor') and self.grandmaster_advisor:
                self.log_text(f"🎯 Preparing AI v2 for {recommended_hero} card evaluation...")
            
            # Update status to show readiness
            self.update_status(f"Hero Selected: {recommended_hero} | Ready for Card Picks")
            
            self.log_text(f"✅ System ready for hero-aware card recommendations")
            
            # Update progression display
            if hasattr(self, 'update_draft_progression_display'):
                self.update_draft_progression_display()
            
        except Exception as e:
            self.log_text(f"⚠️ Error preparing for card draft: {e}")
    
    def _show_system_health(self):
        """Show current AI v2 system health status."""
        try:
            if not self.ai_v2_integrator:
                return
                
            health = self.ai_v2_integrator.check_system_health()
            overall_status = health.get('overall_status', 'unknown')
            
            self.log_text(f"🔧 AI v2 System Health: {overall_status}")
            
            # Show component status
            components = health.get('components', {})
            for component, status in components.items():
                if isinstance(status, dict):
                    comp_status = status.get('status', 'unknown')
                    response_time = status.get('response_time_ms', 0)
                    self.log_text(f"   • {component}: {comp_status} ({response_time:.0f}ms)")
            
            # Show data source status
            data_sources = health.get('data_sources', {})
            if 'hsreplay' in data_sources:
                hsreplay_status = data_sources['hsreplay'].get('status', 'unknown')
                self.log_text(f"   • HSReplay API: {hsreplay_status}")
                
        except Exception as e:
            self.log_text(f"⚠️ Could not check system health: {e}")
    
    def _show_draft_summary(self):
        """Show comprehensive draft summary with hero and AI v2 context."""
        try:
            self.log_text(f"\n{'📊' * 50}")
            self.log_text("📊 DRAFT SUMMARY")
            
            if self.selected_hero_class:
                self.log_text(f"📊 Hero: {self.selected_hero_class}")
                
                if self.hero_recommendation:
                    confidence = self.hero_recommendation.confidence_level
                    self.log_text(f"📊 Hero Confidence: {confidence:.1%}")
            
            self.log_text(f"📊 Total Cards Drafted: {self.draft_picks_count}")
            
            # Show AI v2 system performance
            if self.ai_v2_integrator:
                try:
                    performance = self.ai_v2_integrator._get_performance_summary()
                    hero_recs = performance.get('hero_recommendations_last_hour', 0)
                    card_evals = performance.get('card_evaluations_last_hour', 0)
                    
                    self.log_text(f"📊 AI v2 Performance:")
                    self.log_text(f"   • Hero recommendations: {hero_recs}")
                    self.log_text(f"   • Card evaluations: {card_evals}")
                except:
                    pass
            
            self.log_text(f"{'📊' * 50}")
            
        except Exception as e:
            self.log_text(f"⚠️ Error showing draft summary: {e}")
    
    def _display_hero_selection_ui(self, hero_recommendation):
        """Display dedicated hero selection panel with winrate displays, confidence indicators, and qualitative descriptions."""
        try:
            # Create hero selection window
            hero_window = tk.Toplevel(self.root)
            hero_window.title("👑 AI v2 Hero Selection - Grandmaster Coach")
            hero_window.geometry("900x700")
            hero_window.configure(bg='#1a1a2e')
            hero_window.attributes('-topmost', True)
            
            # Make it modal
            hero_window.transient(self.root)
            hero_window.grab_set()
            
            # Title section
            title_frame = tk.Frame(hero_window, bg='#1a1a2e')
            title_frame.pack(fill='x', padx=20, pady=10)
            
            title_label = tk.Label(title_frame, 
                                 text="👑 HERO SELECTION - AI v2 GRANDMASTER COACH",
                                 font=('Arial', 16, 'bold'), fg='#ffd700', bg='#1a1a2e')
            title_label.pack()
            
            subtitle_label = tk.Label(title_frame, 
                                    text="Statistical Analysis + Meta Insights + Qualitative Assessment",
                                    font=('Arial', 10), fg='#cccccc', bg='#1a1a2e')
            subtitle_label.pack()
            
            # Confidence indicator
            confidence_frame = tk.Frame(hero_window, bg='#1a1a2e')
            confidence_frame.pack(fill='x', padx=20, pady=5)
            
            confidence_text = f"📊 Analysis Confidence: {hero_recommendation.confidence_level:.1%}"
            confidence_color = '#27ae60' if hero_recommendation.confidence_level > 0.7 else '#f39c12' if hero_recommendation.confidence_level > 0.5 else '#e74c3c'
            
            confidence_label = tk.Label(confidence_frame, text=confidence_text,
                                      font=('Arial', 11, 'bold'), fg=confidence_color, bg='#1a1a2e')
            confidence_label.pack()
            
            # Heroes analysis section
            heroes_frame = tk.Frame(hero_window, bg='#1a1a2e')
            heroes_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            for i, hero_analysis in enumerate(hero_recommendation.hero_analysis):
                hero_class = hero_analysis['class']
                winrate = hero_analysis['winrate']
                profile = hero_analysis.get('profile', {})
                explanation = hero_analysis.get('explanation', '')
                is_recommended = (i == hero_recommendation.recommended_hero_index)
                
                # Hero card frame
                hero_frame = tk.Frame(heroes_frame, 
                                    bg='#ffd700' if is_recommended else '#16213e',
                                    relief='raised' if is_recommended else 'flat',
                                    bd=3 if is_recommended else 1)
                hero_frame.pack(fill='x', pady=8)
                
                # Hero header
                header_frame = tk.Frame(hero_frame, bg='#ffd700' if is_recommended else '#16213e')
                header_frame.pack(fill='x', padx=10, pady=5)
                
                # Hero name and recommendation indicator
                hero_title = f"{'👑 ' if is_recommended else ''}#{i+1}. {hero_class}"
                if is_recommended:
                    hero_title += " - RECOMMENDED"
                
                hero_label = tk.Label(header_frame, text=hero_title,
                                    font=('Arial', 14, 'bold'), 
                                    fg='#1a1a2e' if is_recommended else '#ffd700',
                                    bg='#ffd700' if is_recommended else '#16213e')
                hero_label.pack(side='left')
                
                # Winrate display
                winrate_text = f"{winrate:.1f}%"
                winrate_color = '#27ae60' if winrate > 52 else '#f39c12' if winrate > 48 else '#e74c3c'
                winrate_label = tk.Label(header_frame, text=winrate_text,
                                       font=('Arial', 12, 'bold'), 
                                       fg=winrate_color,
                                       bg='#ffd700' if is_recommended else '#16213e')
                winrate_label.pack(side='right')
                
                # Hero details
                details_frame = tk.Frame(hero_frame, bg='#ffd700' if is_recommended else '#16213e')
                details_frame.pack(fill='x', padx=10, pady=5)
                
                # Qualitative details
                playstyle = profile.get('playstyle', 'Unknown')
                complexity = profile.get('complexity', 'Unknown')
                description = profile.get('description', 'No description available')
                
                details_text = f"Playstyle: {playstyle} | Complexity: {complexity}\n{description}"
                details_label = tk.Label(details_frame, text=details_text,
                                       font=('Arial', 9), 
                                       fg='#1a1a2e' if is_recommended else '#cccccc',
                                       bg='#ffd700' if is_recommended else '#16213e',
                                       wraplength=800, justify='left')
                details_label.pack(anchor='w')
                
                # AI explanation
                if explanation:
                    explanation_label = tk.Label(details_frame, text=f"Analysis: {explanation}",
                                                font=('Arial', 9, 'italic'), 
                                                fg='#1a1a2e' if is_recommended else '#cccccc',
                                                bg='#ffd700' if is_recommended else '#16213e',
                                                wraplength=800, justify='left')
                    explanation_label.pack(anchor='w', pady=(5, 0))
            
            # Overall recommendation section
            recommendation_frame = tk.Frame(hero_window, bg='#1a1a2e')
            recommendation_frame.pack(fill='x', padx=20, pady=10)
            
            recommendation_text = hero_recommendation.explanation
            recommendation_label = tk.Label(recommendation_frame, text=f"🎯 Overall Analysis:\n{recommendation_text}",
                                          font=('Arial', 10), fg='#ffd700', bg='#1a1a2e',
                                          wraplength=850, justify='left')
            recommendation_label.pack()
            
            # Action buttons
            button_frame = tk.Frame(hero_window, bg='#1a1a2e')
            button_frame.pack(fill='x', padx=20, pady=10)
            
            close_button = tk.Button(button_frame, text="✅ Continue to Card Draft",
                                   font=('Arial', 12, 'bold'), 
                                   bg='#27ae60', fg='white',
                                   command=hero_window.destroy)
            close_button.pack(side='right', padx=10)
            
            health_button = tk.Button(button_frame, text="📊 System Health",
                                    font=('Arial', 10), 
                                    bg='#3498db', fg='white',
                                    command=lambda: self._show_system_health())
            health_button.pack(side='left', padx=10)
            
            # Store current hero for context
            recommended_hero = hero_recommendation.hero_classes[hero_recommendation.recommended_hero_index]
            self.current_hero = recommended_hero
            
            self.log_text(f"👑 Hero selection UI displayed for {len(hero_recommendation.hero_classes)} heroes")
            
        except Exception as e:
            self.log_text(f"❌ Error displaying hero selection UI: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_system_health(self):
        """Show system health status for AI v2 components."""
        try:
            if not self.ai_v2_integrator:
                messagebox.showinfo("System Health", "AI v2 system not available")
                return
            
            health = self.ai_v2_integrator.check_system_health()
            
            # Create health display window
            health_window = tk.Toplevel(self.root)
            health_window.title("📊 AI v2 System Health")
            health_window.geometry("600x500")
            health_window.configure(bg='#2c3e50')
            health_window.attributes('-topmost', True)
            
            # Health text area
            health_text = scrolledtext.ScrolledText(health_window, 
                                                  font=('Consolas', 10),
                                                  bg='#34495e', fg='#ecf0f1')
            health_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Format health information
            health_info = f"🔍 AI v2 SYSTEM HEALTH REPORT\n"
            health_info += f"{'=' * 50}\n\n"
            health_info += f"Overall Status: {health['overall_status'].value.upper()}\n"
            health_info += f"System Uptime: {health['uptime_hours']:.1f} hours\n\n"
            
            health_info += "📊 COMPONENT STATUS:\n"
            for component, status in health['components'].items():
                status_emoji = "✅" if status['status'] == 'online' else "⚠️" if status['status'] == 'degraded' else "❌"
                health_info += f"{status_emoji} {component}: {status['status']}\n"
                if 'response_time_ms' in status:
                    health_info += f"   Response time: {status['response_time_ms']:.1f}ms\n"
                if status.get('fallback_active'):
                    health_info += f"   🔄 Fallback mode active\n"
            
            health_info += "\n📡 DATA SOURCES:\n"
            for source, info in health['data_sources'].items():
                if isinstance(info, dict):
                    source_emoji = "✅" if info.get('status') == 'online' else "⚠️"
                    health_info += f"{source_emoji} {source}: {info.get('status', 'unknown')}\n"
            
            health_info += f"\n⚡ PERFORMANCE:\n"
            perf = health['performance']
            health_info += f"Hero recommendations: {perf['hero_recommendations_last_hour']}/hour\n"
            health_info += f"Card evaluations: {perf['card_evaluations_last_hour']}/hour\n"
            health_info += f"Avg hero response: {perf['avg_hero_response_time_ms']:.1f}ms\n"
            health_info += f"Avg card eval: {perf['avg_card_eval_time_ms']:.1f}ms\n"
            
            health_text.insert('1.0', health_info)
            health_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("System Health Error", f"Failed to get system health: {e}")
    
    def toggle_ultimate_detection(self, *args):
        """Toggle Ultimate Detection Engine on/off with immediate feedback."""
        if not self.ultimate_detector:
            self.log_text("⚠️ Ultimate Detection Engine not available.")
            self.use_ultimate_detection.set(False)
            return

        if self.use_ultimate_detection.get():
            self.log_text("🚀 Ultimate Detection Engine ENABLED - Using advanced analysis.")
            if not self.cache_build_in_progress:
                threading.Thread(target=self._load_ultimate_database, daemon=True).start()
        else:
            self.log_text("📉 Ultimate Detection Engine DISABLED - Using basic histogram matching.")
    
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
    
    def toggle_phash_detection(self, *args):
        """Toggle pHash Detection on/off with immediate feedback."""
        if not self.phash_matcher:
            self.log_text("⚠️ pHash Detection not available. Install 'imagehash'.")
            self.use_phash_detection.set(False)
            return

        if self.use_phash_detection.get():
            self.log_text("⚡ pHash Detection ENABLED - Using ultra-fast pre-filtering.")
        else:
            self.log_text("📉 pHash Detection DISABLED - Using standard detection methods.")
    
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
    
    def open_advanced_correction_center(self):
        """Open the advanced manual correction center for both hero and card corrections."""
        try:
            correction_window = tk.Toplevel(self.root)
            correction_window.title("🔧 Advanced Manual Correction Center")
            correction_window.geometry("1000x800")
            correction_window.configure(bg='#2C3E50')
            correction_window.attributes('-topmost', True)
            
            # Make it modal
            correction_window.transient(self.root)
            correction_window.grab_set()
            
            # Title section
            title_frame = tk.Frame(correction_window, bg='#34495E', relief='raised', bd=2)
            title_frame.pack(fill='x', padx=10, pady=5)
            
            title_label = tk.Label(
                title_frame,
                text="🔧 ADVANCED MANUAL CORRECTION CENTER",
                font=('Arial', 16, 'bold'),
                fg='#E74C3C',
                bg='#34495E'
            )
            title_label.pack(pady=10)
            
            subtitle_label = tk.Label(
                title_frame,
                text="Comprehensive Correction Workflow for Hero and Card AI Decisions",
                font=('Arial', 10),
                fg='#BDC3C7',
                bg='#34495E'
            )
            subtitle_label.pack()
            
            # Main content area with tabs
            from tkinter import ttk
            style = ttk.Style()
            style.theme_use('clam')
            
            notebook = ttk.Notebook(correction_window)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Hero Correction Tab
            hero_tab = tk.Frame(notebook, bg='#34495E')
            notebook.add(hero_tab, text="👑 Hero Corrections")
            self._create_hero_correction_tab(hero_tab)
            
            # Card Correction Tab
            card_tab = tk.Frame(notebook, bg='#34495E')
            notebook.add(card_tab, text="🃏 Card Corrections")
            self._create_card_correction_tab(card_tab)
            
            # System Override Tab
            override_tab = tk.Frame(notebook, bg='#34495E')
            notebook.add(override_tab, text="⚙️ System Overrides")
            self._create_system_override_tab(override_tab)
            
            # History Tab
            history_tab = tk.Frame(notebook, bg='#34495E')
            notebook.add(history_tab, text="📋 Correction History")
            self._create_correction_history_tab(history_tab)
            
            # Control buttons
            controls_frame = tk.Frame(correction_window, bg='#2C3E50')
            controls_frame.pack(fill='x', padx=10, pady=5)
            
            # Apply All button
            apply_btn = tk.Button(
                controls_frame,
                text="✅ Apply All Corrections",
                command=lambda: self._apply_all_corrections(correction_window),
                bg='#27AE60',
                fg='white',
                font=('Arial', 11, 'bold'),
                relief='raised',
                bd=3
            )
            apply_btn.pack(side='left', padx=5)
            
            # Reset button
            reset_btn = tk.Button(
                controls_frame,
                text="🔄 Reset All",
                command=lambda: self._reset_all_corrections(correction_window),
                bg='#F39C12',
                fg='white',
                font=('Arial', 10),
                relief='raised',
                bd=2
            )
            reset_btn.pack(side='left', padx=5)
            
            # Close button
            close_btn = tk.Button(
                controls_frame,
                text="❌ Close",
                command=correction_window.destroy,
                bg='#E74C3C',
                fg='white',
                font=('Arial', 10),
                relief='raised',
                bd=2
            )
            close_btn.pack(side='right', padx=5)
            
        except Exception as e:
            self.log_text(f"❌ Error opening correction center: {e}")
            messagebox.showerror("Error", f"Failed to open correction center: {e}")
    
    def _create_hero_correction_tab(self, parent):
        """Create the hero correction tab for manual hero choice overrides."""
        # Header
        header_frame = tk.Frame(parent, bg='#34495E')
        header_frame.pack(fill='x', padx=10, pady=10)
        
        header_label = tk.Label(
            header_frame,
            text="👑 Manual Hero Selection Override",
            font=('Arial', 14, 'bold'),
            fg='#F39C12',
            bg='#34495E'
        )
        header_label.pack()
        
        # Instructions
        instructions_label = tk.Label(
            header_frame,
            text="Override AI hero recommendations when you disagree with the analysis",
            font=('Arial', 10),
            fg='#BDC3C7',
            bg='#34495E'
        )
        instructions_label.pack()
        
        # Current hero status
        status_frame = tk.Frame(parent, bg='#34495E', relief='sunken', bd=2)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        current_hero_text = f"Current Hero: {self.current_hero or 'None Selected'}"
        self.current_hero_label = tk.Label(
            status_frame,
            text=current_hero_text,
            font=('Arial', 12, 'bold'),
            fg='#E74C3C',
            bg='#34495E'
        )
        self.current_hero_label.pack(pady=5)
        
        # Hero override section
        override_frame = tk.Frame(parent, bg='#34495E')
        override_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Hero selection dropdown
        hero_selection_frame = tk.Frame(override_frame, bg='#34495E')
        hero_selection_frame.pack(fill='x', pady=10)
        
        tk.Label(
            hero_selection_frame,
            text="Select Hero to Override:",
            font=('Arial', 11, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(side='left')
        
        hero_classes = ['WARRIOR', 'PALADIN', 'HUNTER', 'ROGUE', 'PRIEST', 
                       'SHAMAN', 'MAGE', 'WARLOCK', 'DRUID', 'DEMONHUNTER']
        
        self.hero_override_var = tk.StringVar(value="Select Hero...")
        hero_dropdown = ttk.Combobox(
            hero_selection_frame,
            textvariable=self.hero_override_var,
            values=hero_classes,
            state='readonly',
            width=15
        )
        hero_dropdown.pack(side='left', padx=10)
        
        # Override reason text area
        reason_frame = tk.Frame(override_frame, bg='#34495E')
        reason_frame.pack(fill='both', expand=True, pady=10)
        
        tk.Label(
            reason_frame,
            text="Reason for Override (optional):",
            font=('Arial', 11, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(anchor='w')
        
        self.hero_override_reason = scrolledtext.ScrolledText(
            reason_frame,
            height=8,
            width=80,
            font=('Arial', 10),
            bg='#16213E',
            fg='#ECF0F1',
            insertbackground='#ECF0F1'
        )
        self.hero_override_reason.pack(fill='both', expand=True, pady=5)
        self.hero_override_reason.insert('1.0', "Enter your reasoning for overriding the AI's hero recommendation...")
        
        # Apply hero override button
        apply_hero_btn = tk.Button(
            override_frame,
            text="👑 Apply Hero Override",
            command=self._apply_hero_override,
            bg='#8E44AD',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3
        )
        apply_hero_btn.pack(pady=10)
    
    def _create_card_correction_tab(self, parent):
        """Create the card correction tab for manual card choice overrides."""
        # Header
        header_frame = tk.Frame(parent, bg='#34495E')
        header_frame.pack(fill='x', padx=10, pady=10)
        
        header_label = tk.Label(
            header_frame,
            text="🃏 Manual Card Selection Override",
            font=('Arial', 14, 'bold'),
            fg='#3498DB',
            bg='#34495E'
        )
        header_label.pack()
        
        # Instructions
        instructions_label = tk.Label(
            header_frame,
            text="Override AI card recommendations and provide feedback for learning",
            font=('Arial', 10),
            fg='#BDC3C7',
            bg='#34495E'
        )
        instructions_label.pack()
        
        # Last recommendation display
        last_rec_frame = tk.Frame(parent, bg='#34495E', relief='sunken', bd=2)
        last_rec_frame.pack(fill='x', padx=10, pady=5)
        
        self.last_recommendation_label = tk.Label(
            last_rec_frame,
            text="Last AI Recommendation: No recent recommendations",
            font=('Arial', 11),
            fg='#E74C3C',
            bg='#34495E'
        )
        self.last_recommendation_label.pack(pady=5)
        
        # Card override section
        override_frame = tk.Frame(parent, bg='#34495E')
        override_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Manual card search
        search_frame = tk.Frame(override_frame, bg='#34495E')
        search_frame.pack(fill='x', pady=10)
        
        tk.Label(
            search_frame,
            text="Override with Card:",
            font=('Arial', 11, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(side='left')
        
        self.card_override_entry = tk.Entry(
            search_frame,
            font=('Arial', 11),
            width=30,
            bg='#16213E',
            fg='#ECF0F1',
            insertbackground='#ECF0F1'
        )
        self.card_override_entry.pack(side='left', padx=10)
        self.card_override_entry.bind('<KeyRelease>', self._update_card_suggestions)
        
        # Card suggestions listbox
        suggestions_frame = tk.Frame(override_frame, bg='#34495E')
        suggestions_frame.pack(fill='x', pady=5)
        
        tk.Label(
            suggestions_frame,
            text="Card Suggestions:",
            font=('Arial', 10, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(anchor='w')
        
        self.card_suggestions_listbox = tk.Listbox(
            suggestions_frame,
            height=6,
            font=('Arial', 10),
            bg='#16213E',
            fg='#ECF0F1',
            selectbackground='#3498DB'
        )
        self.card_suggestions_listbox.pack(fill='x', pady=5)
        self.card_suggestions_listbox.bind('<Double-Button-1>', self._select_suggestion)
        
        # Override feedback area
        feedback_frame = tk.Frame(override_frame, bg='#34495E')
        feedback_frame.pack(fill='both', expand=True, pady=10)
        
        tk.Label(
            feedback_frame,
            text="Feedback for AI Learning:",
            font=('Arial', 11, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(anchor='w')
        
        self.card_override_feedback = scrolledtext.ScrolledText(
            feedback_frame,
            height=6,
            width=80,
            font=('Arial', 10),
            bg='#16213E',
            fg='#ECF0F1',
            insertbackground='#ECF0F1'
        )
        self.card_override_feedback.pack(fill='both', expand=True, pady=5)
        self.card_override_feedback.insert('1.0', "Explain why you chose this card over the AI's recommendation...")
        
        # Apply card override button
        apply_card_btn = tk.Button(
            override_frame,
            text="🃏 Apply Card Override",
            command=self._apply_card_override,
            bg='#27AE60',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3
        )
        apply_card_btn.pack(pady=10)
    
    def _create_hero_correction_tab(self, parent):
        """Create the hero correction tab for manual hero choice overrides."""
        # Header
        header_frame = tk.Frame(parent, bg='#34495E')
        header_frame.pack(fill='x', padx=10, pady=10)
        
        header_label = tk.Label(
            header_frame,
            text="👑 Manual Hero Selection Override",
            font=('Arial', 14, 'bold'),
            fg='#F39C12',
            bg='#34495E'
        )
        header_label.pack()
        
        # Instructions
        instructions_label = tk.Label(
            header_frame,
            text="Override AI hero recommendations when you disagree with the analysis",
            font=('Arial', 10),
            fg='#BDC3C7',
            bg='#34495E'
        )
        instructions_label.pack()
        
        # Current hero status
        status_frame = tk.Frame(parent, bg='#34495E', relief='sunken', bd=2)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        current_hero_text = f"Current Hero: {getattr(self, 'current_hero', None) or 'None Selected'}"
        self.current_hero_label = tk.Label(
            status_frame,
            text=current_hero_text,
            font=('Arial', 12, 'bold'),
            fg='#E74C3C',
            bg='#34495E'
        )
        self.current_hero_label.pack(pady=5)
        
        # Hero override section
        override_frame = tk.Frame(parent, bg='#34495E')
        override_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Hero selection dropdown
        hero_selection_frame = tk.Frame(override_frame, bg='#34495E')
        hero_selection_frame.pack(fill='x', pady=10)
        
        tk.Label(
            hero_selection_frame,
            text="Select Hero to Override:",
            font=('Arial', 11, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(side='left')
        
        hero_classes = ['WARRIOR', 'PALADIN', 'HUNTER', 'ROGUE', 'PRIEST', 
                       'SHAMAN', 'MAGE', 'WARLOCK', 'DRUID', 'DEMONHUNTER']
        
        self.hero_override_var = tk.StringVar(value="Select Hero...")
        from tkinter import ttk
        hero_dropdown = ttk.Combobox(
            hero_selection_frame,
            textvariable=self.hero_override_var,
            values=hero_classes,
            state='readonly',
            width=15
        )
        hero_dropdown.pack(side='left', padx=10)
        
        # Override reason text area
        reason_frame = tk.Frame(override_frame, bg='#34495E')
        reason_frame.pack(fill='both', expand=True, pady=10)
        
        tk.Label(
            reason_frame,
            text="Reason for Override (optional):",
            font=('Arial', 11, 'bold'),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(anchor='w')
        
        self.hero_override_reason = scrolledtext.ScrolledText(
            reason_frame,
            height=8,
            width=80,
            font=('Arial', 10),
            bg='#16213E',
            fg='#ECF0F1',
            insertbackground='#ECF0F1'
        )
        self.hero_override_reason.pack(fill='both', expand=True, pady=5)
        self.hero_override_reason.insert('1.0', "Enter your reasoning for overriding the AI's hero recommendation...")
        
        # Apply hero override button
        apply_hero_btn = tk.Button(
            override_frame,
            text="👑 Apply Hero Override",
            command=self._apply_hero_override,
            bg='#8E44AD',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3
        )
        apply_hero_btn.pack(pady=10)
    
    def _apply_hero_override(self):
        """Apply the hero override selection."""
        try:
            selected_hero = self.hero_override_var.get()
            if selected_hero == "Select Hero...":
                messagebox.showwarning("No Selection", "Please select a hero to override to.")
                return
            
            reason = self.hero_override_reason.get('1.0', 'end-1c').strip()
            if reason == "Enter your reasoning for overriding the AI's hero recommendation...":
                reason = "Manual override (no reason provided)"
            
            # Update the current hero
            self.current_hero = selected_hero
            if hasattr(self, 'current_hero_label'):
                self.current_hero_label.config(text=f"Current Hero: {selected_hero}")
            
            # Log the override
            self.log_text(f"🔄 Hero override applied: {selected_hero}")
            self.log_text(f"📝 Reason: {reason}")
            
            messagebox.showinfo("Override Applied", f"Hero override applied successfully!\nNew Hero: {selected_hero}")
            
        except Exception as e:
            self.log_text(f"❌ Error applying hero override: {str(e)}")
            messagebox.showerror("Error", f"Failed to apply hero override: {str(e)}")
    
    def _create_system_override_tab(self, parent):
        """Create the system override tab for granular AI control."""
        # Header
        header_frame = tk.Frame(parent, bg='#34495E')
        header_frame.pack(fill='x', padx=10, pady=10)
        
        header_label = tk.Label(
            header_frame,
            text="⚙️ System Override Controls",
            font=('Arial', 14, 'bold'),
            fg='#E67E22',
            bg='#34495E'
        )
        header_label.pack()
        
        # Emergency controls section
        emergency_frame = tk.LabelFrame(
            parent,
            text="Emergency Fallback Controls",
            font=('Arial', 12, 'bold'),
            fg='#E74C3C',
            bg='#34495E',
            bd=2,
            relief='groove'
        )
        emergency_frame.pack(fill='x', padx=10, pady=10)
        
        # AI system toggles
        self.hero_ai_enabled = tk.BooleanVar(value=True)
        hero_toggle = tk.Checkbutton(
            emergency_frame,
            text="🤖 Hero AI System Enabled",
            variable=self.hero_ai_enabled,
            font=('Arial', 11),
            fg='#ECF0F1',
            bg='#34495E',
            selectcolor='#27AE60'
        )
        hero_toggle.pack(anchor='w', padx=10, pady=5)
        
        self.card_ai_enabled = tk.BooleanVar(value=True)
        card_toggle = tk.Checkbutton(
            emergency_frame,
            text="🤖 Card AI System Enabled",
            variable=self.card_ai_enabled,
            font=('Arial', 11),
            fg='#ECF0F1',
            bg='#34495E',
            selectcolor='#27AE60'
        )
        card_toggle.pack(anchor='w', padx=10, pady=5)
        
        # Data source controls
        data_frame = tk.LabelFrame(
            parent,
            text="Data Source Controls",
            font=('Arial', 12, 'bold'),
            fg='#3498DB',
            bg='#34495E',
            bd=2,
            relief='groove'
        )
        data_frame.pack(fill='x', padx=10, pady=10)
        
        self.hsreplay_enabled = tk.BooleanVar(value=True)
        hsreplay_toggle = tk.Checkbutton(
            data_frame,
            text="📊 HSReplay Data Integration",
            variable=self.hsreplay_enabled,
            font=('Arial', 11),
            fg='#ECF0F1',
            bg='#34495E',
            selectcolor='#27AE60'
        )
        hsreplay_toggle.pack(anchor='w', padx=10, pady=5)
        
        # Confidence thresholds
        threshold_frame = tk.LabelFrame(
            parent,
            text="Confidence Thresholds",
            font=('Arial', 12, 'bold'),
            fg='#8E44AD',
            bg='#34495E',
            bd=2,
            relief='groove'
        )
        threshold_frame.pack(fill='x', padx=10, pady=10)
        
        # Hero confidence threshold
        hero_threshold_frame = tk.Frame(threshold_frame, bg='#34495E')
        hero_threshold_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(
            hero_threshold_frame,
            text="Hero Recommendation Threshold:",
            font=('Arial', 10),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(side='left')
        
        self.hero_confidence_threshold = tk.DoubleVar(value=0.6)
        hero_scale = tk.Scale(
            hero_threshold_frame,
            variable=self.hero_confidence_threshold,
            from_=0.1,
            to=0.9,
            resolution=0.1,
            orient='horizontal',
            length=300,
            bg='#34495E',
            fg='#ECF0F1'
        )
        hero_scale.pack(side='right')
        
        # Card confidence threshold
        card_threshold_frame = tk.Frame(threshold_frame, bg='#34495E')
        card_threshold_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(
            card_threshold_frame,
            text="Card Recommendation Threshold:",
            font=('Arial', 10),
            fg='#ECF0F1',
            bg='#34495E'
        ).pack(side='left')
        
        self.card_confidence_threshold = tk.DoubleVar(value=0.7)
        card_scale = tk.Scale(
            card_threshold_frame,
            variable=self.card_confidence_threshold,
            from_=0.1,
            to=0.9,
            resolution=0.1,
            orient='horizontal',
            length=300,
            bg='#34495E',
            fg='#ECF0F1'
        )
        card_scale.pack(side='right')
    
    def _create_correction_history_tab(self, parent):
        """Create the correction history tab to show past overrides."""
        # Header
        header_frame = tk.Frame(parent, bg='#34495E')
        header_frame.pack(fill='x', padx=10, pady=10)
        
        header_label = tk.Label(
            header_frame,
            text="📋 Correction History & Learning",
            font=('Arial', 14, 'bold'),
            fg='#27AE60',
            bg='#34495E'
        )
        header_label.pack()
        
        # History display
        history_text = scrolledtext.ScrolledText(
            parent,
            height=25,
            width=100,
            font=('Consolas', 9),
            bg='#16213E',
            fg='#ECF0F1',
            insertbackground='#ECF0F1'
        )
        history_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Load correction history
        history_content = self._load_correction_history()
        history_text.insert('1.0', history_content)
        history_text.config(state='disabled')
    
    def _apply_hero_override(self):
        """Apply the manual hero override."""
        try:
            selected_hero = self.hero_override_var.get()
            reason = self.hero_override_reason.get('1.0', 'end-1c')
            
            if selected_hero == "Select Hero...":
                messagebox.showwarning("Warning", "Please select a hero to override")
                return
            
            # Update current hero
            self.current_hero = selected_hero
            
            # Log the override
            self._log_correction("HERO", f"Override: {selected_hero}", reason)
            
            # Update UI
            self.current_hero_label.config(text=f"Current Hero: {selected_hero}")
            
            # Show confirmation
            messagebox.showinfo("Success", f"Hero override applied: {selected_hero}")
            
        except Exception as e:
            self.log_text(f"❌ Error applying hero override: {e}")
            messagebox.showerror("Error", f"Failed to apply hero override: {e}")
    
    def _apply_card_override(self):
        """Apply the manual card override."""
        try:
            card_name = self.card_override_entry.get()
            feedback = self.card_override_feedback.get('1.0', 'end-1c')
            
            if not card_name:
                messagebox.showwarning("Warning", "Please enter a card name")
                return
            
            # Log the override
            self._log_correction("CARD", f"Override: {card_name}", feedback)
            
            # Update last recommendation display
            self.last_recommendation_label.config(text=f"Last Override: {card_name}")
            
            # Show confirmation
            messagebox.showinfo("Success", f"Card override applied: {card_name}")
            
        except Exception as e:
            self.log_text(f"❌ Error applying card override: {e}")
            messagebox.showerror("Error", f"Failed to apply card override: {e}")
    
    def _update_card_suggestions(self, event=None):
        """Update card suggestions based on user input."""
        try:
            search_text = self.card_override_entry.get().lower()
            if len(search_text) < 2:
                self.card_suggestions_listbox.delete(0, 'end')
                return
            
            # Get card suggestions from cards database
            suggestions = []
            if hasattr(self, 'cards_json_loader') and self.cards_json_loader:
                all_cards = self.cards_json_loader.get_all_card_names()
                suggestions = [name for name in all_cards if search_text in name.lower()][:10]
            
            # Update listbox
            self.card_suggestions_listbox.delete(0, 'end')
            for suggestion in suggestions:
                self.card_suggestions_listbox.insert('end', suggestion)
                
        except Exception as e:
            self.log_text(f"❌ Error updating suggestions: {e}")
    
    def _select_suggestion(self, event=None):
        """Select a card from suggestions."""
        try:
            selection = self.card_suggestions_listbox.curselection()
            if selection:
                selected_card = self.card_suggestions_listbox.get(selection[0])
                self.card_override_entry.delete(0, 'end')
                self.card_override_entry.insert(0, selected_card)
        except Exception as e:
            self.log_text(f"❌ Error selecting suggestion: {e}")
    
    def _apply_all_corrections(self, window):
        """Apply all pending corrections."""
        try:
            # Apply system settings
            corrections_applied = 0
            
            # Log system settings changes
            self._log_correction("SYSTEM", "Settings Updated", "System override settings applied")
            corrections_applied += 1
            
            messagebox.showinfo("Success", f"Applied {corrections_applied} corrections successfully")
            window.destroy()
            
        except Exception as e:
            self.log_text(f"❌ Error applying corrections: {e}")
            messagebox.showerror("Error", f"Failed to apply corrections: {e}")
    
    def _reset_all_corrections(self, window):
        """Reset all correction settings to defaults."""
        try:
            # Reset system toggles
            self.hero_ai_enabled.set(True)
            self.card_ai_enabled.set(True)
            self.hsreplay_enabled.set(True)
            
            # Reset thresholds
            self.hero_confidence_threshold.set(0.6)
            self.card_confidence_threshold.set(0.7)
            
            # Clear override fields
            self.hero_override_var.set("Select Hero...")
            self.hero_override_reason.delete('1.0', 'end')
            self.hero_override_reason.insert('1.0', "Enter your reasoning for overriding the AI's hero recommendation...")
            
            self.card_override_entry.delete(0, 'end')
            self.card_override_feedback.delete('1.0', 'end')
            self.card_override_feedback.insert('1.0', "Explain why you chose this card over the AI's recommendation...")
            
            messagebox.showinfo("Reset", "All correction settings reset to defaults")
            
        except Exception as e:
            self.log_text(f"❌ Error resetting corrections: {e}")
            messagebox.showerror("Error", f"Failed to reset corrections: {e}")
    
    def _log_correction(self, correction_type, action, reason):
        """Log a manual correction for learning purposes."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {correction_type}: {action} | Reason: {reason}\n"
            
            # Append to correction log file
            log_file = Path("corrections_log.txt")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            # Also log to main text area
            self.log_text(f"📝 Correction logged: {correction_type} - {action}")
            
        except Exception as e:
            self.log_text(f"❌ Error logging correction: {e}")
    
    def _load_correction_history(self):
        """Load and format correction history."""
        try:
            log_file = Path("corrections_log.txt")
            if not log_file.exists():
                return "No correction history available yet.\n\nThis area will show your manual overrides and feedback to help the AI learn from your decisions."
            
            with open(log_file, 'r', encoding='utf-8') as f:
                history = f.read()
                
            if not history.strip():
                return "No correction history available yet.\n\nThis area will show your manual overrides and feedback to help the AI learn from your decisions."
                
            return f"Recent Corrections:\n{'='*50}\n\n{history}"
            
        except Exception as e:
            return f"Error loading correction history: {e}"


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