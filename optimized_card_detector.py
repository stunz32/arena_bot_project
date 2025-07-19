#!/usr/bin/env python3
"""
Optimized Card Detection System
Uses a smaller, focused database for better matching accuracy.
"""

import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

class OptimizedCardDetector:
    """
    Optimized card detector using focused database for higher accuracy.
    """
    
    def __init__(self, max_database_size: int = 1000):
        """Initialize with smaller, focused database."""
        self.logger = logging.getLogger(__name__)
        self.max_database_size = max_database_size
        
        # Initialize components
        self.smart_detector = None
        self.histogram_matcher = None
        self.cards_loader = None
        self.asset_loader = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all detection components."""
        try:
            from arena_bot.core.smart_coordinate_detector import get_smart_coordinate_detector
            from arena_bot.detection.histogram_matcher import get_histogram_matcher
            from arena_bot.data.cards_json_loader import get_cards_json_loader
            from arena_bot.utils.asset_loader import get_asset_loader
            
            self.smart_detector = get_smart_coordinate_detector()
            self.histogram_matcher = get_histogram_matcher()
            self.cards_loader = get_cards_json_loader()
            self.asset_loader = get_asset_loader()
            
            self.logger.info("Loading optimized card database...")
            
            # Load focused card database
            self._load_focused_database()
            
            self.logger.info("✅ Optimized card detector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _load_focused_database(self):
        """Load a smaller, more focused card database."""
        self.logger.info("Loading card list from JSON...")
        cards = self.cards_loader.load_cards_json()
        
        # Filter to current draftable cards only
        current_cards = [card for card in cards if card.get('collectible', False)]
        self.logger.info(f"Filtered {len(cards)} cards -> {len(current_cards)} collectible cards")
        
        # Get available card codes
        card_codes = self.asset_loader.get_available_card_codes()
        self.logger.info(f"Found {len(card_codes)} available card codes")
        
        # Get draftable cards
        draftable_cards = self.cards_loader.get_draftable_cards()
        self.logger.info(f"Filtered {len(card_codes)} cards -> {len(draftable_cards)} draftable cards")
        
        # Limit database size for better performance and accuracy
        if len(draftable_cards) > self.max_database_size:
            # Prioritize recent sets and common arena cards
            priority_sets = ['TITANS', 'TOY', 'ULD', 'CORE', 'CLASSIC']
            
            priority_cards = []
            other_cards = []
            
            for card_code in draftable_cards:
                card_data = next((c for c in current_cards if c.get('dbfId') == self.asset_loader.get_card_id(card_code)), None)
                if card_data:
                    card_set = card_data.get('set', '')
                    if any(p in card_set for p in priority_sets):
                        priority_cards.append(card_code)
                    else:
                        other_cards.append(card_code)
            
            # Take priority cards first, then fill with others
            selected_cards = priority_cards[:self.max_database_size]
            remaining_slots = self.max_database_size - len(selected_cards)
            if remaining_slots > 0:
                selected_cards.extend(other_cards[:remaining_slots])
                
            draftable_cards = selected_cards
            
        self.logger.info(f"Loading focused database with {len(draftable_cards)} cards")
        
        # Load histograms for selected cards
        self.histogram_matcher.load_card_histograms(draftable_cards)
        
        self.logger.info(f"✅ Loaded {len(draftable_cards)} card images for optimized detection")
    
    def detect_cards(self, screenshot_path: str) -> List[Dict[str, Any]]:
        """Detect cards with optimized matching."""
        self.logger.info(f"📸 Analyzing screenshot: {screenshot_path}")
        
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            raise ValueError(f"Could not load screenshot: {screenshot_path}")
        
        height, width = screenshot.shape[:2]
        self.logger.info(f"📸 Analyzing screenshot: {screenshot_path} ({width}x{height})")
        
        self.logger.info("🎯 Starting optimized card detection")
        
        # 1. Smart coordinate detection
        interface_coords, card_positions = self.smart_detector.detect_interface_and_cards(screenshot)
        
        if not interface_coords or not card_positions:
            self.logger.error("❌ Failed to detect interface or card positions")
            return []
        
        self.logger.info(f"✅ Interface detected: {interface_coords}")
        self.logger.info(f"✅ Card positions: {len(card_positions)} cards")
        
        # 2. Detect each card
        results = []
        for i, (x, y, w, h) in enumerate(card_positions, 1):
            self.logger.info(f"🔍 Processing card {i} at ({x}, {y}, {w}, {h})")
            
            # Extract card region
            card_region = screenshot[y:y+h, x:x+w]
            
            # Multiple detection strategies
            best_match = self._detect_single_card(card_region, f"card_{i}")
            
            if best_match:
                best_match['position'] = (x, y, w, h)
                results.append(best_match)
                self.logger.info(f"✅ Card {i}: {best_match['name']} (conf: {best_match['confidence']:.3f}, strategy: {best_match['strategy']})")
            else:
                self.logger.warning(f"❌ Failed to identify card {i}")
        
        self.logger.info(f"🎉 Optimized detection complete: {len(results)}/{len(card_positions)} cards identified")
        
        return results
    
    def _detect_single_card(self, card_region: np.ndarray, debug_name: str = "card") -> Optional[Dict[str, Any]]:
        """Detect a single card using multiple strategies."""
        
        strategies = [
            ("full_card", self._process_full_card),
            ("center_crop", self._process_center_crop),
            ("arena_tracker_80x80", self._process_arena_tracker_style),
            ("upper_70_percent", self._process_upper_portion)
        ]
        
        results = []
        
        for strategy_name, processor in strategies:
            try:
                processed_region = processor(card_region)
                matches = self.histogram_matcher.find_best_matches(processed_region, top_k=5)
                
                if matches:
                    best_match = matches[0]
                    results.append({
                        'strategy': strategy_name,
                        'card_code': best_match['card_code'],
                        'name': best_match['name'],
                        'confidence': best_match['confidence'],
                        'distance': best_match['distance']
                    })
                    
            except Exception as e:
                self.logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        if not results:
            return None
        
        # Weighted scoring - prioritize strategies that work best
        strategy_weights = {
            'full_card': 1.0,
            'center_crop': 0.9,
            'arena_tracker_80x80': 0.8,
            'upper_70_percent': 0.7
        }
        
        for result in results:
            weight = strategy_weights.get(result['strategy'], 0.5)
            result['weighted_confidence'] = result['confidence'] * weight
        
        # Return best weighted result
        best_result = max(results, key=lambda x: x['weighted_confidence'])
        
        # Save debug image
        self._save_debug_image(card_region, f"optimized_{debug_name}.png")
        
        return best_result
    
    def _process_full_card(self, card_region: np.ndarray) -> np.ndarray:
        """Process full card region."""
        # Resize to Arena Tracker standard
        return cv2.resize(card_region, (80, 80))
    
    def _process_center_crop(self, card_region: np.ndarray) -> np.ndarray:
        """Process center-cropped region."""
        h, w = card_region.shape[:2]
        # Take center 70% of card
        crop_h, crop_w = int(h * 0.7), int(w * 0.7)
        start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = card_region[start_y:start_y+crop_h, start_x:start_x+crop_w]
        return cv2.resize(cropped, (80, 80))
    
    def _process_arena_tracker_style(self, card_region: np.ndarray) -> np.ndarray:
        """Process using Arena Tracker's exact methodology."""
        return cv2.resize(card_region, (80, 80))
    
    def _process_upper_portion(self, card_region: np.ndarray) -> np.ndarray:
        """Process upper 70% of card (focus on artwork)."""
        h, w = card_region.shape[:2]
        upper_region = card_region[:int(h * 0.7), :]
        return cv2.resize(upper_region, (80, 80))
    
    def _save_debug_image(self, image: np.ndarray, filename: str):
        """Save debug image."""
        try:
            cv2.imwrite(filename, image)
        except Exception as e:
            self.logger.warning(f"Could not save debug image {filename}: {e}")

def main():
    """Main detection function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 OPTIMIZED CARD DETECTION SYSTEM")
    print("=" * 80)
    print("🎯 Using focused database for improved accuracy!")
    print("✅ Smart coordinate detection + optimized matching")
    print("✅ Smaller database = higher accuracy + faster performance")
    print("=" * 80)
    print()
    
    screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
    
    print(f"🔍 Analyzing: {screenshot_path}")
    print()
    
    try:
        detector = OptimizedCardDetector(max_database_size=1000)  # Use 1000 cards max
        results = detector.detect_cards(screenshot_path)
        
        print("=" * 80)
        print("🎯 OPTIMIZED DETECTION RESULTS")
        print("=" * 80)
        
        if results:
            print(f"✅ SUCCESS: {len(results)}/3 cards detected")
            print(f"📊 Accuracy: {len(results)/3*100:.1f}%")
            
            # Show interface info
            detector_instance = detector.smart_detector
            screenshot = cv2.imread(screenshot_path)
            interface_coords, _ = detector_instance.detect_interface_and_cards(screenshot)
            print(f"🎮 Interface: {interface_coords}")
            print()
            
            for i, result in enumerate(results, 1):
                print(f"📋 Card {i}: {result['name']}")
                print(f"   Code: {result['card_code']}")
                print(f"   Confidence: {result['confidence']:.3f} | Weighted: {result['weighted_confidence']:.3f}")
                print(f"   Strategy: {result['strategy']}")
                print(f"   Position: {result['position']}")
                print()
            
            # Verify target cards
            target_cards = {
                1: "Clay Matriarch",
                2: "Dwarven Archaeologist", 
                3: "Cyclopean Crusher"
            }
            
            print("🎯 TARGET CARD VERIFICATION:")
            for i, result in enumerate(results, 1):
                expected = target_cards.get(i, "Unknown")
                actual = result['name']
                status = "✅" if expected.lower() in actual.lower() or actual.lower() in expected.lower() else "❌"
                print(f"{status} Card {i}: Expected {expected}, got {actual}")
                
        else:
            print("❌ No cards detected")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()