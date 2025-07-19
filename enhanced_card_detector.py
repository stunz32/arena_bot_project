#!/usr/bin/env python3
"""
Enhanced Card Detection System
Combines smart coordinate detection with robust card identification.
This is the ingenious solution for reliable detection on all the RIGHT cards.
"""

import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))

# Import debug image manager
from debug_image_manager import DebugImageManager

class EnhancedCardDetector:
    """
    The complete solution for reliable card detection:
    1. Smart coordinate detection (finds interface automatically)
    2. Multiple region extraction strategies
    3. Arena Tracker-style histogram matching
    4. Confidence scoring and validation
    """
    
    def __init__(self):
        """Initialize the enhanced card detector."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.smart_detector = None
        self.histogram_matcher = None
        self.cards_loader = None
        self.asset_loader = None
        
        # Initialize debug image manager
        self.debug_manager = DebugImageManager()
        
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
            
            # Load card database for detection
            self._load_enhanced_database()
            
            self.logger.info("✅ Enhanced card detector initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _load_enhanced_database(self):
        """Load card database with Arena Tracker methodology."""
        try:
            self.logger.info("Loading enhanced card database...")
            
            # Get available cards
            available_cards = self.asset_loader.get_available_cards()
            
            # Filter to collectible, draftable cards only
            draftable_cards = []
            for card_code in available_cards:
                # Skip non-draftable card types
                if any(card_code.startswith(prefix) for prefix in 
                       ['HERO_', 'BG_', 'TB_', 'KARA_', 'PVPDR_']):
                    continue
                
                # Check if card is collectible via JSON database
                if self.cards_loader and self.cards_loader.is_collectible(card_code):
                    draftable_cards.append(card_code)
                elif not self.cards_loader:
                    # Fallback: include all non-excluded cards
                    draftable_cards.append(card_code)
            
            self.logger.info(f"Filtered {len(available_cards)} cards -> {len(draftable_cards)} draftable cards")
            
            # Load card images and compute histograms
            card_images = {}
            for card_code in draftable_cards:
                for is_premium in [False, True]:
                    try:
                        image = cv2.imread(str(self.asset_loader.assets_dir / "cards" / f"{card_code}{'_premium' if is_premium else ''}.png"))
                        if image is not None:
                            hist_key = f"{card_code}{'_premium' if is_premium else ''}"
                            card_images[hist_key] = image
                    except Exception:
                        continue
            
            # Load into histogram matcher
            self.histogram_matcher.load_card_database(card_images)
            self.logger.info(f"✅ Loaded {len(card_images)} card images for detection")
            
        except Exception as e:
            self.logger.error(f"Failed to load card database: {e}")
            raise
    
    def _extract_arena_tracker_region(self, card_image: np.ndarray, is_premium: bool = False) -> Optional[np.ndarray]:
        """Extract Arena Tracker's proven 80x80 region."""
        try:
            # Arena Tracker's exact coordinates
            if is_premium:
                x, y, w, h = 57, 71, 80, 80
            else:
                x, y, w, h = 60, 71, 80, 80
            
            # Check bounds
            if (card_image.shape[1] < x + w) or (card_image.shape[0] < y + h):
                # Fallback: resize and try again
                if card_image.shape[0] >= 80 and card_image.shape[1] >= 80:
                    resized = cv2.resize(card_image, (218, 300), interpolation=cv2.INTER_AREA)
                    if is_premium:
                        return resized[71:151, 57:137]
                    else:
                        return resized[71:151, 60:140]
                return None
            
            return card_image[y:y+h, x:x+w]
            
        except Exception as e:
            self.logger.error(f"Error extracting AT region: {e}")
            return None
    
    def _compute_arena_tracker_histogram(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Compute Arena Tracker's exact histogram."""
        try:
            if image is None or image.size == 0:
                return None
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Arena Tracker's exact parameters
            h_bins = 50
            s_bins = 60
            hist_size = [h_bins, s_bins]
            ranges = [0, 180, 0, 256]
            channels = [0, 1]
            
            # Calculate histogram
            hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error computing histogram: {e}")
            return None
    
    def _process_card_region(self, card_image: np.ndarray, position: int) -> List[Dict[str, Any]]:
        """
        Process a card region using multiple extraction strategies.
        Returns multiple candidates with confidence scores.
        """
        strategies = []
        
        try:
            h, w = card_image.shape[:2]
            
            # Strategy 1: Full card resized (proven most effective in debug)
            full_resized = cv2.resize(card_image, (80, 80), interpolation=cv2.INTER_AREA)
            strategies.append(("full_card", full_resized))
            
            # Strategy 2: Center crop (second most effective)
            if h >= 60 and w >= 60:
                center_crop = card_image[30:h-30, 30:w-30]
                if center_crop.size > 0:
                    # Resize to 80x80 for comparison
                    center_resized = cv2.resize(center_crop, (80, 80), interpolation=cv2.INTER_AREA)
                    strategies.append(("center_crop_80x80", center_resized))
            
            # Strategy 3: Arena Tracker exact region (traditional method)
            at_region = self._extract_arena_tracker_region(card_image, is_premium=False)
            if at_region is not None:
                strategies.append(("arena_tracker_80x80", at_region))
            
            # Strategy 4: Upper 70% (card art focus)
            upper_region = card_image[0:int(h*0.7), :]
            if upper_region.size > 0:
                upper_resized = cv2.resize(upper_region, (80, 80), interpolation=cv2.INTER_AREA)
                strategies.append(("upper_70_percent", upper_resized))
            
            # Strategy 5: Card art region (20px border crop)
            if h >= 40 and w >= 40:
                art_region = card_image[20:h-20, 20:w-20]
                if art_region.size > 0:
                    art_resized = cv2.resize(art_region, (80, 80), interpolation=cv2.INTER_AREA)
                    strategies.append(("art_region", art_resized))
            
        except Exception as e:
            self.logger.error(f"Error preparing strategies for card {position}: {e}")
            return []
        
        # Process each strategy
        candidates = []
        for strategy_name, processed_image in strategies:
            try:
                # Compute histogram
                hist = self._compute_arena_tracker_histogram(processed_image)
                if hist is None:
                    continue
                
                # Find best matches
                matches = self.histogram_matcher.find_best_matches(hist, max_candidates=3)
                
                for match in matches:
                    candidates.append({
                        'card_code': match.card_code,
                        'confidence': match.confidence,
                        'distance': match.distance,
                        'strategy': strategy_name,
                        'position': position
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing strategy {strategy_name}: {e}")
                continue
        
        return candidates
    
    def _select_best_candidate(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best candidate using weighted scoring."""
        if not candidates:
            return None
        
        # Weight strategies by proven effectiveness (based on debug results)
        strategy_weights = {
            'arena_tracker_80x80': 1.0,    # Highest weight - proven method
            'center_crop_80x80': 0.95,     # Very effective in tests
            'full_card': 0.9,              # Actually very good for extracted regions
            'upper_70_percent': 0.8,       # Good for card art
            'art_region': 0.7,             # Decent fallback
        }
        
        # Calculate weighted scores
        for candidate in candidates:
            strategy = candidate['strategy']
            weight = strategy_weights.get(strategy, 0.5)
            
            # Weighted confidence score
            candidate['weighted_score'] = candidate['confidence'] * weight
        
        # Sort by weighted score
        candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Return best candidate if it meets minimum threshold
        best = candidates[0]
        # Lower threshold since our debug shows good matches at 0.15-0.25 range
        if best['weighted_score'] >= 0.12:  # More lenient threshold
            return best
        
        return None
    
    def detect_arena_cards(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        The main detection method - combines smart coordinates with robust identification.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            Dict with complete detection results
        """
        try:
            self.logger.info("🎯 Starting enhanced card detection")
            
            # Step 1: Smart coordinate detection
            coord_result = self.smart_detector.detect_cards_automatically(screenshot)
            if not coord_result or not coord_result['success']:
                self.logger.error("❌ Smart coordinate detection failed")
                return {'success': False, 'error': 'coordinate_detection_failed'}
            
            interface_rect = coord_result['interface_rect']
            card_positions = coord_result['card_positions']
            
            self.logger.info(f"✅ Interface detected: {interface_rect}")
            self.logger.info(f"✅ Card positions: {len(card_positions)} cards")
            
            # Step 2: Extract and identify each card
            detected_cards = []
            
            for i, (x, y, w, h) in enumerate(card_positions):
                self.logger.info(f"🔍 Processing card {i+1} at ({x}, {y}, {w}, {h})")
                
                # Extract card region
                card_image = screenshot[y:y+h, x:x+w]
                
                if card_image.size == 0:
                    self.logger.warning(f"⚠️ Empty region for card {i+1}")
                    continue
                
                # Save debug image using organized system
                self.debug_manager.save_debug_image(
                    card_image, 
                    f"enhanced_card_{i+1}", 
                    'cards'
                )
                
                # Process with multiple strategies
                candidates = self._process_card_region(card_image, i+1)
                
                # Select best candidate
                best_candidate = self._select_best_candidate(candidates)
                
                if best_candidate:
                    # Get card name
                    card_name = self.cards_loader.get_card_name(best_candidate['card_code'].replace('_premium', ''))
                    
                    detected_card = {
                        'position': i + 1,
                        'card_code': best_candidate['card_code'],
                        'card_name': card_name,
                        'confidence': best_candidate['confidence'],
                        'weighted_score': best_candidate['weighted_score'],
                        'strategy': best_candidate['strategy'],
                        'coordinates': (x, y, w, h)
                    }
                    
                    detected_cards.append(detected_card)
                    
                    self.logger.info(f"✅ Card {i+1}: {card_name} (conf: {best_candidate['confidence']:.3f}, strategy: {best_candidate['strategy']})")
                else:
                    self.logger.warning(f"❌ Could not identify card {i+1}")
            
            # Step 3: Compile final results
            result = {
                'success': len(detected_cards) > 0,
                'interface_rect': interface_rect,
                'card_positions': card_positions,
                'detected_cards': detected_cards,
                'detection_count': len(detected_cards),
                'accuracy': len(detected_cards) / len(card_positions) if card_positions else 0,
                'method': 'enhanced_detector_v1'
            }
            
            self.logger.info(f"🎉 Enhanced detection complete: {len(detected_cards)}/{len(card_positions)} cards identified")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def analyze_screenshot_file(self, screenshot_path: str) -> Dict[str, Any]:
        """Convenience method to analyze a screenshot file."""
        try:
            screenshot = cv2.imread(screenshot_path)
            if screenshot is None:
                return {'success': False, 'error': f'Could not load screenshot: {screenshot_path}'}
            
            self.logger.info(f"📸 Analyzing screenshot: {screenshot_path} ({screenshot.shape[1]}x{screenshot.shape[0]})")
            
            result = self.detect_arena_cards(screenshot)
            
            # Save debug images if successful
            if result.get('success'):
                # Save using organized debug system
                self.debug_manager.save_detection_results(screenshot, result)
                # Also call smart detector's debug images
                self.smart_detector.save_debug_images(screenshot, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing screenshot {screenshot_path}: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Test the enhanced card detector with the user's screenshot."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 ENHANCED CARD DETECTION SYSTEM")
    print("=" * 80)
    print("🎯 Ingenious solution for reliable detection on all the RIGHT cards!")
    print("✅ Smart coordinate detection + Arena Tracker methodology")
    print("✅ Multiple extraction strategies with confidence scoring")
    print("✅ Automatic interface detection with fallback methods")
    print("=" * 80)
    
    try:
        # Initialize detector
        detector = EnhancedCardDetector()
        
        # Test with user's screenshot
        screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
        
        print(f"\n🔍 Analyzing: {screenshot_path}")
        result = detector.analyze_screenshot_file(screenshot_path)
        
        # Display results
        print(f"\n{'='*80}")
        print("🎯 DETECTION RESULTS")
        print(f"{'='*80}")
        
        if result['success']:
            print(f"✅ SUCCESS: {result['detection_count']}/3 cards detected")
            print(f"📊 Accuracy: {result['accuracy']*100:.1f}%")
            print(f"🎮 Interface: {result['interface_rect']}")
            print()
            
            for card in result['detected_cards']:
                print(f"📋 Card {card['position']}: {card['card_name']}")
                print(f"   Code: {card['card_code']}")
                print(f"   Confidence: {card['confidence']:.3f} | Weighted: {card['weighted_score']:.3f}")
                print(f"   Strategy: {card['strategy']}")
                print(f"   Position: {card['coordinates']}")
                print()
            
            # Check for target cards
            target_cards = ["TOY_380", "ULD_309", "TTN_042"]
            target_names = ["Clay Matriarch", "Dwarven Archaeologist", "Cyclopean Crusher"]
            
            print("🎯 TARGET CARD VERIFICATION:")
            for i, (target_code, target_name) in enumerate(zip(target_cards, target_names)):
                found = False
                for card in result['detected_cards']:
                    if card['position'] == i + 1:
                        detected_code = card['card_code'].replace('_premium', '')
                        if detected_code == target_code:
                            print(f"✅ Card {i+1}: {target_name} - CORRECT!")
                            found = True
                        else:
                            print(f"❌ Card {i+1}: Expected {target_name}, got {card['card_name']}")
                            found = True
                        break
                
                if not found:
                    print(f"❓ Card {i+1}: {target_name} - NOT DETECTED")
            
        else:
            print(f"❌ DETECTION FAILED: {result.get('error', 'Unknown error')}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)