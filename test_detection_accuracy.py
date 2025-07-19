#!/usr/bin/env python3
"""
Direct Detection Accuracy Test
Runs comprehensive testing without interactive interface.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time

# Add project modules
sys.path.insert(0, str(Path(__file__).parent))

def test_current_detection():
    """Test current detection accuracy with available debug images."""
    print("🎯 ARENA BOT DETECTION ACCURACY TEST")
    print("=" * 60)
    
    try:
        # Import detection modules
        from arena_bot.core.smart_coordinate_detector import SmartCoordinateDetector
        from arena_bot.detection.histogram_matcher import HistogramMatcher
        from arena_bot.detection.phash_matcher import PerceptualHashMatcher
        
        print("✅ All detection modules imported successfully")
        
        # Initialize detectors
        smart_detector = SmartCoordinateDetector()
        histogram_matcher = HistogramMatcher()
        
        # Load a focused card database for testing
        print("📚 Loading card database for testing...")
        from arena_bot.utils.asset_loader import AssetLoader
        from pathlib import Path
        
        asset_loader = AssetLoader()
        card_images = {}
        
        # Load a subset of cards for testing (faster than full 12K+ database)
        test_cards = ["CS3_001", "AT_001", "AT_002", "SW_001", "SW_002", "YOP_001", "YOP_002", 
                     "DMF_001", "DMF_002", "AV_100", "AV_101", "BAR_020", "BAR_021"]
        
        for card_code in test_cards:
            for is_premium in [False, True]:
                suffix = "_premium" if is_premium else ""
                image = asset_loader.load_card_image(card_code, is_premium)
                if image is not None:
                    card_images[f"{card_code}{suffix}"] = image
        
        histogram_matcher.load_card_database(card_images)
        print(f"✅ Loaded {len(histogram_matcher.card_histograms)} card histograms for testing")
        
        try:
            phash_matcher = PerceptualHashMatcher()
            phash_available = True
            print("✅ pHash matcher available")
        except Exception as e:
            phash_available = False
            print(f"⚠️ pHash matcher unavailable: {e}")
        
        # Test with available debug images
        debug_images = [
            "debug_card_1.png",
            "debug_card_2.png", 
            "debug_card_3.png"
        ]
        
        print("\n📊 TESTING DETECTION ACCURACY:")
        print("-" * 40)
        
        test_results = []
        
        for i, image_name in enumerate(debug_images, 1):
            image_path = Path(__file__).parent / image_name
            
            if not image_path.exists():
                print(f"⚠️ Card {i}: {image_name} not found")
                continue
                
            print(f"\n🔍 Testing Card {i}: {image_name}")
            
            # Load image
            card_image = cv2.imread(str(image_path))
            if card_image is None:
                print(f"❌ Failed to load {image_name}")
                continue
            
            h, w = card_image.shape[:2]
            print(f"   Image size: {w}×{h} pixels")
            
            # Test histogram matching
            start_time = time.time()
            try:
                query_hist = histogram_matcher.compute_histogram(card_image)
                if query_hist is not None:
                    matches = histogram_matcher.find_best_matches(query_hist, max_candidates=3)
                    if matches:
                        best_match = matches[0]
                        hist_time = (time.time() - start_time) * 1000
                        print(f"   📊 Histogram: {best_match.card_code} (conf: {best_match.confidence:.3f}, {hist_time:.1f}ms)")
                        
                        # Store result
                        test_results.append({
                            'card': i,
                            'method': 'histogram',
                            'result': best_match.card_code,
                            'confidence': best_match.confidence,
                            'time_ms': hist_time,
                            'success': best_match.confidence > 0.5
                        })
                    else:
                        print("   📊 Histogram: No matches found")
                else:
                    print("   📊 Histogram: Failed to compute histogram")
            except Exception as e:
                print(f"   📊 Histogram: Error - {e}")
            
            # Test pHash if available
            if phash_available:
                start_time = time.time()
                try:
                    phash_result = phash_matcher.find_best_phash_match(card_image, confidence_threshold=0.6)
                    phash_time = (time.time() - start_time) * 1000
                    
                    if phash_result:
                        print(f"   ⚡ pHash: {phash_result.card_code} (conf: {phash_result.confidence:.3f}, {phash_time:.1f}ms)")
                        
                        test_results.append({
                            'card': i,
                            'method': 'phash',
                            'result': phash_result.card_code,
                            'confidence': phash_result.confidence,
                            'time_ms': phash_time,
                            'success': phash_result.confidence > 0.6
                        })
                    else:
                        print(f"   ⚡ pHash: No confident match ({phash_time:.1f}ms)")
                except Exception as e:
                    print(f"   ⚡ pHash: Error - {e}")
        
        # Generate summary report
        print("\n📈 DETECTION SUMMARY:")
        print("-" * 40)
        
        if test_results:
            histogram_results = [r for r in test_results if r['method'] == 'histogram']
            phash_results = [r for r in test_results if r['method'] == 'phash']
            
            if histogram_results:
                hist_success_rate = sum(1 for r in histogram_results if r['success']) / len(histogram_results)
                avg_hist_time = np.mean([r['time_ms'] for r in histogram_results])
                avg_hist_conf = np.mean([r['confidence'] for r in histogram_results])
                
                print(f"📊 Histogram Matching:")
                print(f"   Success Rate: {hist_success_rate:.1%}")
                print(f"   Average Time: {avg_hist_time:.1f}ms")
                print(f"   Average Confidence: {avg_hist_conf:.3f}")
            
            if phash_results:
                phash_success_rate = sum(1 for r in phash_results if r['success']) / len(phash_results)
                avg_phash_time = np.mean([r['time_ms'] for r in phash_results])
                avg_phash_conf = np.mean([r['confidence'] for r in phash_results])
                
                print(f"⚡ pHash Matching:")
                print(f"   Success Rate: {phash_success_rate:.1%}")
                print(f"   Average Time: {avg_phash_time:.1f}ms")
                print(f"   Average Confidence: {avg_phash_conf:.3f}")
            
            # Performance assessment
            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            print("-" * 40)
            
            total_tests = len([r for r in test_results if r['success']])
            total_possible = len(debug_images) * 2 if phash_available else len(debug_images)
            
            if total_possible > 0:
                overall_success = total_tests / total_possible
                print(f"Overall Success Rate: {overall_success:.1%}")
                
                if overall_success >= 0.9:
                    print("✅ EXCELLENT: Detection accuracy exceeds 90%")
                elif overall_success >= 0.7:
                    print("🟡 GOOD: Detection accuracy above 70%")
                elif overall_success >= 0.5:
                    print("🟠 FAIR: Detection accuracy above 50%")
                else:
                    print("🔴 NEEDS IMPROVEMENT: Detection accuracy below 50%")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_coordinate_detection():
    """Test the enhanced SmartCoordinateDetector."""
    print("\n🎯 SMART COORDINATE DETECTOR TEST")
    print("=" * 60)
    
    try:
        # Load a test screenshot if available
        test_image_path = Path(__file__).parent / "debug_card_2.png"
        
        if test_image_path.exists():
            print(f"📸 Testing with: {test_image_path.name}")
            
            # Create a mock screenshot (expand the card image to simulate full screen)
            card_image = cv2.imread(str(test_image_path))
            if card_image is not None:
                # Create a realistic mock arena interface with proper scaling
                mock_screenshot = np.zeros((1440, 3440, 3), dtype=np.uint8)
                
                # Arena interface coordinates (based on checkpoint documentation)
                interface_x, interface_y = 1122, 233  # Arena interface position
                interface_w, interface_h = 1197, 704  # Arena interface size
                
                # Card positions within interface (proper arena card size: 447×493)
                card_w_arena, card_h_arena = 447, 493
                card_positions = [
                    (interface_x + 80, interface_y + 100),   # Left card
                    (interface_x + 375, interface_y + 100),  # Middle card  
                    (interface_x + 670, interface_y + 100)   # Right card
                ]
                
                # Scale and place card at proper arena position
                card_resized = cv2.resize(card_image, (card_w_arena, card_h_arena), interpolation=cv2.INTER_AREA)
                
                # Place resized card in center position
                pos_x, pos_y = card_positions[1]  # Middle position
                end_x = min(pos_x + card_w_arena, 3440)
                end_y = min(pos_y + card_h_arena, 1440)
                
                mock_screenshot[pos_y:end_y, pos_x:end_x] = card_resized[:end_y-pos_y, :end_x-pos_x]
                
                # Add arena background color (dark brown/red) to make interface detectable
                interface_area = mock_screenshot[interface_y:interface_y+interface_h, interface_x:interface_x+interface_w]
                interface_area[interface_area.sum(axis=2) == 0] = [45, 25, 15]  # Dark brown background
                
                print(f"   Created mock arena interface: {interface_w}×{interface_h}")
                print(f"   Card placed at: ({pos_x}, {pos_y}) with size {card_w_arena}×{card_h_arena}")
                
                # Test SmartCoordinateDetector
                from arena_bot.core.smart_coordinate_detector import SmartCoordinateDetector
                detector = SmartCoordinateDetector()
                
                start_time = time.time()
                result = detector.detect_cards_automatically(mock_screenshot)
                detection_time = (time.time() - start_time) * 1000
                
                if result and result.get('success'):
                    print(f"✅ Smart detection successful ({detection_time:.1f}ms)")
                    print(f"   Method: {result.get('detection_method', 'unknown')}")
                    print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
                    print(f"   Cards detected: {len(result.get('card_positions', []))}")
                    
                    # Show card size used
                    card_size = result.get('card_size_used', (0, 0))
                    print(f"   Dynamic card size: {card_size[0]}×{card_size[1]} pixels")
                    
                    # Show optimization info if available
                    if result.get('optimization_available'):
                        stats = result.get('stats', {})
                        print(f"   pHash-ready regions: {stats.get('phash_ready_regions', 0)}")
                        print(f"   Method confidence: {result.get('method_confidence', 0.0):.3f}")
                        
                        recommended_methods = stats.get('recommended_methods', [])
                        print(f"   Recommended methods: {recommended_methods}")
                    
                    return True
                else:
                    confidence = result.get('confidence', 0.0) if result else 0.0
                    print(f"❌ Smart detection failed (confidence: {confidence:.3f})")
                    return False
            else:
                print("❌ Failed to load test image")
                return False
        else:
            print("⚠️ No test image available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing coordinate detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Arena Bot Accuracy Testing...")
    
    # Test detection accuracy
    detection_results = test_current_detection()
    
    # Test coordinate detection  
    coordinate_success = analyze_coordinate_detection()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL ASSESSMENT")
    print("=" * 60)
    
    if detection_results:
        successful_detections = sum(1 for r in detection_results if r['success'])
        total_detections = len(detection_results)
        success_rate = successful_detections / total_detections if total_detections > 0 else 0
        
        print(f"Detection Success Rate: {success_rate:.1%} ({successful_detections}/{total_detections})")
    
    if coordinate_success:
        print("✅ Smart Coordinate Detection: WORKING")
    else:
        print("🔴 Smart Coordinate Detection: NEEDS ATTENTION")
    
    print("\n🎯 Ready for further improvements!")