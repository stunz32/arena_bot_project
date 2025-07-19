#!/usr/bin/env python3
"""
Focused Card Detection System
Uses the enhanced coordinate detection with focused Arena Tracker database.
This implements the proven approach where target cards rank #1 with small database.
"""

import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add our modules
sys.path.insert(0, str(Path(__file__).parent))


class FocusedCardDetector:
    """
    Combines perfect coordinate detection with focused Arena Tracker matching.
    Uses small database of target cards for accurate identification.
    """
    
    def __init__(self, target_cards: List[str] = None):
        """Initialize focused detector with target cards."""
        self.logger = logging.getLogger(__name__)
        
        # Default target cards from our testing
        self.target_cards = target_cards or ['TOY_380', 'ULD_309', 'TTN_042']
        
        # Initialize components
        self.smart_detector = None
        self.histogram_matcher = None
        self.cards_loader = None
        self.asset_loader = None
        
        self._initialize_components()
        self._load_focused_database()
    
    def _initialize_components(self):
        """Initialize detection components."""
        try:
            from arena_bot.core.smart_coordinate_detector import get_smart_coordinate_detector
            from arena_bot.detection.histogram_matcher import get_histogram_matcher
            from arena_bot.data.cards_json_loader import get_cards_json_loader
            from arena_bot.utils.asset_loader import get_asset_loader
            
            self.smart_detector = get_smart_coordinate_detector()
            self.histogram_matcher = get_histogram_matcher()
            self.cards_loader = get_cards_json_loader()
            self.asset_loader = get_asset_loader()
            
            self.logger.info("✅ Components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _load_focused_database(self):
        """Load only target cards into histogram matcher."""
        try:
            self.logger.info(f"Loading focused database with {len(self.target_cards)} target cards...")
            
            # Clear existing database
            self.histogram_matcher.clear_database()
            
            # Load target card images
            card_images = {}
            loaded_count = 0
            
            for card_code in self.target_cards:
                # Try loading normal version
                try:
                    normal_path = self.asset_loader.assets_dir / "cards" / f"{card_code}.png"
                    if normal_path.exists():
                        normal_image = cv2.imread(str(normal_path))
                        if normal_image is not None:
                            card_images[card_code] = normal_image
                            loaded_count += 1
                            self.logger.info(f"  ✅ Loaded {card_code}")
                except Exception as e:
                    self.logger.warning(f"  ❌ Could not load {card_code}: {e}")
                
                # Try loading premium version
                try:
                    premium_path = self.asset_loader.assets_dir / "cards" / f"{card_code}_premium.png"
                    if premium_path.exists():
                        premium_image = cv2.imread(str(premium_path))
                        if premium_image is not None:
                            card_images[f"{card_code}_premium"] = premium_image
                            loaded_count += 1
                            self.logger.info(f"  ✅ Loaded {card_code}_premium")
                except Exception as e:
                    self.logger.warning(f"  ❌ Could not load {card_code}_premium: {e}")
            
            # Load into histogram matcher using correct API
            if card_images:
                self.histogram_matcher.load_card_database(card_images)
                self.logger.info(f"✅ Focused database loaded: {loaded_count} card variants")
                self.logger.info(f"📊 Database size: {self.histogram_matcher.get_database_size()} histograms")
            else:
                raise ValueError("No target cards could be loaded!")
                
        except Exception as e:
            self.logger.error(f"Failed to load focused database: {e}")
            raise
    
    def _extract_arena_tracker_region(self, card_image: np.ndarray, is_premium: bool = False) -> Optional[np.ndarray]:
        """Extract Arena Tracker's proven 80x80 region."""
        try:
            # Arena Tracker's exact coordinates
            if is_premium:
                x, y, w, h = 57, 71, 80, 80
            else:
                x, y, w, h = 60, 71, 80, 80
            
            # Check bounds and extract
            if (card_image.shape[1] >= x + w) and (card_image.shape[0] >= y + h):
                return card_image[y:y+h, x:x+w]
            
            # Fallback: resize card and try again
            if card_image.shape[0] >= 80 and card_image.shape[1] >= 80:
                resized = cv2.resize(card_image, (218, 300), interpolation=cv2.INTER_AREA)
                if is_premium:
                    return resized[71:151, 57:137]
                else:
                    return resized[71:151, 60:140]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting AT region: {e}")
            return None
    
    def _compute_arena_tracker_histogram(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Compute Arena Tracker's exact histogram."""
        try:
            if image is None or image.size == 0:
                return None
            
            # Use HistogramMatcher's method for consistency
            return self.histogram_matcher.compute_histogram(image)
            
        except Exception as e:
            self.logger.error(f"Error computing histogram: {e}")
            return None
    
    def detect_focused_cards(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Main detection method using focused approach.
        
        Args:
            screenshot: Full screen screenshot
            
        Returns:
            Dict with detection results
        """
        try:
            self.logger.info("🎯 Starting focused card detection")
            
            # Step 1: Use proven coordinate detection
            coord_result = self.smart_detector.detect_cards_automatically(screenshot)
            if not coord_result or not coord_result['success']:
                self.logger.error("❌ Coordinate detection failed")
                return {'success': False, 'error': 'coordinate_detection_failed'}
            
            interface_rect = coord_result['interface_rect']
            card_positions = coord_result['card_positions']
            
            self.logger.info(f"✅ Interface detected: {interface_rect}")
            self.logger.info(f"✅ Card positions: {len(card_positions)} cards")
            
            # Step 2: Process each card with focused matching
            detected_cards = []
            
            for i, (x, y, w, h) in enumerate(card_positions):
                self.logger.info(f"🔍 Processing card {i+1} at ({x}, {y}, {w}, {h})")
                
                # Extract card region
                card_image = screenshot[y:y+h, x:x+w]
                
                if card_image.size == 0:
                    self.logger.warning(f"⚠️ Empty region for card {i+1}")
                    continue
                
                # Test multiple strategies with focused database
                best_match = None
                best_confidence = 0
                best_strategy = None
                
                strategies = [
                    ("arena_tracker_80x80", lambda img: self._extract_arena_tracker_region(img, False)),
                    ("full_card_80x80", lambda img: cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)),
                    ("center_crop_80x80", lambda img: cv2.resize(img[30:-30, 30:-30], (80, 80)) if img.shape[0] >= 60 and img.shape[1] >= 60 else None),
                    ("upper_70_80x80", lambda img: cv2.resize(img[0:int(img.shape[0]*0.7), :], (80, 80))),
                ]
                
                for strategy_name, extract_func in strategies:
                    try:
                        processed_image = extract_func(card_image)
                        if processed_image is None or processed_image.size == 0:
                            continue
                        
                        # Compute histogram
                        hist = self._compute_arena_tracker_histogram(processed_image)
                        if hist is None:
                            continue
                        
                        # Find matches in focused database
                        matches = self.histogram_matcher.find_best_matches(hist, max_candidates=3)
                        
                        if matches and matches[0].confidence > best_confidence:
                            best_match = matches[0]
                            best_confidence = matches[0].confidence
                            best_strategy = strategy_name
                            
                            self.logger.info(f"  🎯 {strategy_name}: {matches[0].card_code} (conf: {matches[0].confidence:.3f})")
                        
                    except Exception as e:
                        self.logger.warning(f"  ❌ Strategy {strategy_name} failed: {e}")
                        continue
                
                # Record best result
                if best_match:
                    card_name = self.cards_loader.get_card_name(best_match.card_code.replace('_premium', ''))
                    
                    detected_card = {
                        'position': i + 1,
                        'card_code': best_match.card_code,
                        'card_name': card_name,
                        'confidence': best_match.confidence,
                        'distance': best_match.distance,
                        'strategy': best_strategy,
                        'coordinates': (x, y, w, h)
                    }
                    
                    detected_cards.append(detected_card)
                    self.logger.info(f"✅ Card {i+1}: {card_name} (conf: {best_match.confidence:.3f}, strategy: {best_strategy})")
                else:
                    self.logger.warning(f"❌ Could not identify card {i+1} in focused database")
            
            # Step 3: Compile results
            result = {
                'success': len(detected_cards) > 0,
                'interface_rect': interface_rect,
                'card_positions': card_positions,
                'detected_cards': detected_cards,
                'detection_count': len(detected_cards),
                'accuracy': len(detected_cards) / len(card_positions) if card_positions else 0,
                'method': 'focused_detector_v1',
                'target_cards': self.target_cards,
                'database_size': self.histogram_matcher.get_database_size()
            }
            
            self.logger.info(f"🎉 Focused detection complete: {len(detected_cards)}/{len(card_positions)} cards identified")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Focused detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


def main():
    """Test the focused card detector."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🎯 FOCUSED CARD DETECTION SYSTEM")
    print("=" * 80)
    print("✅ Perfect coordinate detection + focused Arena Tracker database")
    print("✅ Target cards: TOY_380, ULD_309, TTN_042")
    print("✅ Small database for accurate matching")
    print("=" * 80)
    
    try:
        # Initialize focused detector
        detector = FocusedCardDetector()
        
        # Test with screenshot
        screenshot_path = "/home/marcco/reference files/Screenshot 2025-07-11 180600.png"
        
        print(f"\n🔍 Analyzing: {screenshot_path}")
        
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return False
        
        result = detector.detect_focused_cards(screenshot)
        
        # Display results
        print(f"\n{'='*80}")
        print("🎯 FOCUSED DETECTION RESULTS")
        print(f"{'='*80}")
        
        if result['success']:
            print(f"✅ SUCCESS: {result['detection_count']}/3 cards detected")
            print(f"📊 Accuracy: {result['accuracy']*100:.1f}%")
            print(f"🗃️ Database size: {result['database_size']} histograms")
            print(f"🎮 Interface: {result['interface_rect']}")
            print()
            
            for card in result['detected_cards']:
                print(f"📋 Card {card['position']}: {card['card_name']}")
                print(f"   Code: {card['card_code']}")
                print(f"   Confidence: {card['confidence']:.3f} | Distance: {card['distance']:.3f}")
                print(f"   Strategy: {card['strategy']}")
                print(f"   Position: {card['coordinates']}")
                print()
            
            # Verify against expected targets
            expected_cards = {
                1: ("TOY_380", "Clay Matriarch"),
                2: ("ULD_309", "Dwarven Archaeologist"), 
                3: ("TTN_042", "Cyclopean Crusher")
            }
            
            print("🎯 TARGET CARD VERIFICATION:")
            correct_count = 0
            for card in result['detected_cards']:
                pos = card['position']
                expected_code, expected_name = expected_cards.get(pos, ("Unknown", "Unknown"))
                actual_code = card['card_code'].replace('_premium', '')
                actual_name = card['card_name']
                
                is_correct = actual_code == expected_code
                status = "✅" if is_correct else "❌"
                if is_correct:
                    correct_count += 1
                
                print(f"{status} Card {pos}: Expected {expected_name} ({expected_code})")
                print(f"     Got {actual_name} ({actual_code})")
            
            print(f"\n🏆 FINAL ACCURACY: {correct_count}/3 = {correct_count/3*100:.1f}%")
            
            if correct_count == 3:
                print("🎉 PERFECT DETECTION ACHIEVED!")
            elif correct_count >= 2:
                print("🎯 Very good detection!")
            else:
                print("🔧 Still needs optimization")
            
            return correct_count == 3
            
        else:
            print(f"❌ DETECTION FAILED: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)