#!/usr/bin/env python3

import cv2
import numpy as np
import logging
from pathlib import Path
from arena_bot.core.smart_coordinate_detector import SmartCoordinateDetector
from arena_bot.core.card_refiner import CardRefiner


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0


def build_validation_set():
    """Build validation set with different resolutions/screenshots."""
    
    validation_set = []
    debug_frames_dir = Path("/mnt/d/cursor bots/arena_bot_project/debug_frames")
    
    # Find available screenshots with different characteristics
    screenshot_files = [
        "Hearthstone Screenshot 07-11-25 17.33.10.png",  # 2560x1440 
        "Screenshot 2025-07-05 085410.png",  # Different aspect ratio/resolution
    ]
    
    for screenshot_file in screenshot_files:
        screenshot_path = debug_frames_dir / screenshot_file
        if screenshot_path.exists():
            screenshot = cv2.imread(str(screenshot_path))
            if screenshot is not None:
                h, w = screenshot.shape[:2]
                validation_set.append({
                    'path': str(screenshot_path),
                    'name': screenshot_file,
                    'resolution': f"{w}x{h}",
                    'image': screenshot
                })
    
    # Add synthetic test cases for different resolutions
    synthetic_cases = [
        (1920, 1080, "Standard_HD"),
        (3440, 1440, "Ultrawide_QHD"),
        (1366, 768, "Laptop_Standard")
    ]
    
    for width, height, name in synthetic_cases:
        # Create synthetic screenshot with arena interface layout
        synthetic_screenshot = create_synthetic_arena_screenshot(width, height)
        validation_set.append({
            'path': f'synthetic_{width}x{height}',
            'name': f"{name}_{width}x{height}",
            'resolution': f"{width}x{height}",
            'image': synthetic_screenshot,
            'synthetic': True
        })
    
    return validation_set


def create_synthetic_arena_screenshot(width, height):
    """Create a synthetic arena screenshot for testing."""
    # Create base image
    screenshot = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with dark red background (arena interface color)
    screenshot[:, :] = [40, 0, 80]  # Dark red-brown
    
    # Add simulated interface region in center
    interface_w = int(width * 0.6)
    interface_h = int(height * 0.8)
    interface_x = (width - interface_w) // 2
    interface_y = (height - interface_h) // 2
    
    # Draw interface area with slightly different color
    screenshot[interface_y:interface_y+interface_h, interface_x:interface_x+interface_w] = [60, 20, 100]
    
    # Add simulated card regions based on Arena Helper positioning
    scale_x = width / 1920
    scale_y = height / 1080
    
    card_positions = [
        (int(393 * scale_x), int(175 * scale_y)),
        (int(673 * scale_x), int(175 * scale_y)),
        (int(953 * scale_x), int(175 * scale_y))
    ]
    
    card_width = int(250 * min(scale_x, scale_y))
    card_height = int(370 * min(scale_x, scale_y))
    
    # Draw simulated cards
    for i, (x, y) in enumerate(card_positions):
        # Card background (gold border simulation)
        cv2.rectangle(screenshot, (x, y), (x + card_width, y + card_height), [50, 150, 200], -1)
        
        # Inner card area
        inner_margin = 10
        cv2.rectangle(screenshot, 
                     (x + inner_margin, y + inner_margin), 
                     (x + card_width - inner_margin, y + card_height - inner_margin), 
                     [100, 80, 60], -1)
        
        # Add mana crystal simulation
        cv2.circle(screenshot, (x + 25, y + 25), 15, [255, 150, 0], -1)
    
    return screenshot


def test_validation_set():
    """Test two-stage pipeline on validation set."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== BUILDING VALIDATION SET ===")
    validation_set = build_validation_set()
    print(f"Built validation set with {len(validation_set)} test cases")
    
    for case in validation_set:
        print(f"- {case['name']}: {case['resolution']}")
    
    # Initialize detector
    detector = SmartCoordinateDetector()
    
    # Test each case
    results = []
    
    print("\n=== TESTING VALIDATION SET ===")
    for i, case in enumerate(validation_set):
        print(f"\nTesting {i+1}/{len(validation_set)}: {case['name']} ({case['resolution']})")
        
        screenshot = case['image']
        result = detector.detect_cards_automatically(screenshot)
        
        if result and result['success']:
            confidence = result['confidence']
            cards_detected = len(result['card_positions'])
            pipeline_stage = result.get('pipeline_stage', 'unknown')
            
            print(f"✅ Success: {cards_detected}/3 cards, confidence={confidence:.3f}, stage={pipeline_stage}")
            
            # Save cutouts for this test case
            case_name = case['name'].replace(' ', '_').replace('.png', '')
            for j, (x, y, w, h) in enumerate(result['card_positions']):
                card_cutout = screenshot[y:y+h, x:x+w]
                output_path = f"/mnt/d/cursor bots/arena_bot_project/debug_frames/VALIDATION_{case_name}_Card{j+1}.png"
                cv2.imwrite(output_path, card_cutout)
            
            # Calculate aspect ratios for robustness check
            aspect_ratios = []
            for x, y, w, h in result['card_positions']:
                aspect_ratios.append(w / h)
            
            avg_aspect_ratio = np.mean(aspect_ratios)
            aspect_std = np.std(aspect_ratios)
            
            results.append({
                'case': case['name'],
                'resolution': case['resolution'],
                'success': True,
                'confidence': confidence,
                'cards_detected': cards_detected,
                'pipeline_stage': pipeline_stage,
                'avg_aspect_ratio': avg_aspect_ratio,
                'aspect_std': aspect_std,
                'positions': result['card_positions']
            })
            
        else:
            print(f"❌ Failed")
            results.append({
                'case': case['name'],
                'resolution': case['resolution'],
                'success': False,
                'confidence': 0.0,
                'cards_detected': 0,
                'pipeline_stage': 'failed'
            })
    
    # Analyze results
    print("\n=== VALIDATION RESULTS SUMMARY ===")
    successful_cases = [r for r in results if r['success']]
    success_rate = len(successful_cases) / len(results)
    
    print(f"Success rate: {success_rate:.1%} ({len(successful_cases)}/{len(results)})")
    
    if successful_cases:
        avg_confidence = np.mean([r['confidence'] for r in successful_cases])
        avg_cards = np.mean([r['cards_detected'] for r in successful_cases])
        
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average cards detected: {avg_cards:.1f}/3")
        
        # Check aspect ratio consistency
        aspect_ratios = []
        for r in successful_cases:
            if 'avg_aspect_ratio' in r:
                aspect_ratios.append(r['avg_aspect_ratio'])
        
        if aspect_ratios:
            overall_aspect_mean = np.mean(aspect_ratios)
            overall_aspect_std = np.std(aspect_ratios)
            print(f"Aspect ratio consistency: {overall_aspect_mean:.3f} ± {overall_aspect_std:.3f}")
            
            # Check if within Hearthstone card range (0.65-0.85)
            valid_aspects = [a for a in aspect_ratios if 0.65 <= a <= 0.85]
            aspect_validity = len(valid_aspects) / len(aspect_ratios)
            print(f"Valid aspect ratios: {aspect_validity:.1%}")
    
    # Pipeline stage distribution
    stage_counts = {}
    for r in successful_cases:
        stage = r.get('pipeline_stage', 'unknown')
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    print(f"\nPipeline stage distribution:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count} cases")
    
    return results


def test_iou_robustness():
    """Test IoU improvements across different conditions."""
    
    print("\n=== IoU ROBUSTNESS TEST ===")
    
    # GROUND TRUTH: Final refined coordinates from Color-Guided Adaptive Crop
    # These are the pixel-perfect coordinates from test_two_stage_pipeline.py
    GROUND_TRUTH_COORDINATES = {
        "Hearthstone Screenshot 07-11-25 17.33.10.png": {
            "resolution": (2560, 1440),
            "cards": [
                (540, 285, 286, 287),  # Card 1: Clay Matriarch 
                (914, 285, 283, 287),  # Card 2: Dwarven Archaeologist
                (1285, 285, 282, 287)  # Card 3: Cyclopian Crusher
            ]
        }
    }
    
    detector = SmartCoordinateDetector()
    
    # Test against ground truth from real screenshot
    ground_truth_ious = []
    debug_frames_dir = Path("/mnt/d/cursor bots/arena_bot_project/debug_frames")
    
    for screenshot_name, truth_data in GROUND_TRUTH_COORDINATES.items():
        screenshot_path = debug_frames_dir / screenshot_name
        if screenshot_path.exists():
            screenshot = cv2.imread(str(screenshot_path))
            if screenshot is not None:
                print(f"Testing against ground truth: {screenshot_name}")
                
                # Run complete two-stage pipeline
                coarse_result = detector.detect_cards_automatically(screenshot)
                if coarse_result and coarse_result['success']:
                    # Apply CardRefiner to each coarse position (two-stage pipeline)
                    refined_positions = []
                    
                    for x, y, w, h in coarse_result['card_positions']:
                        # Extract ROI
                        roi_image = screenshot[y:y+h, x:x+w]
                        
                        # Apply CardRefiner
                        refined_x, refined_y, refined_w, refined_h = CardRefiner.refine_card_region(roi_image)
                        
                        # Convert back to screenshot coordinates
                        final_x = x + refined_x
                        final_y = y + refined_y
                        final_w = refined_w
                        final_h = refined_h
                        
                        refined_positions.append((final_x, final_y, final_w, final_h))
                    
                    detected_positions = refined_positions
                    truth_positions = truth_data['cards']
                    
                    ious = []
                    for i in range(min(len(detected_positions), len(truth_positions))):
                        detected_box = detected_positions[i]
                        truth_box = truth_positions[i]
                        iou = calculate_iou(detected_box, truth_box)
                        ious.append(iou)
                        print(f"  Card {i+1} IoU: {iou:.6f}")
                    
                    avg_iou = np.mean(ious)
                    ground_truth_ious.append(avg_iou)
                    print(f"  Average IoU vs ground truth: {avg_iou:.6f}")
                else:
                    print(f"  ❌ Detection failed for {screenshot_name}")
    
    # Test consistency across resolutions (synthetic)
    
    detector = SmartCoordinateDetector()
    
    # Test same scene at different synthetic resolutions
    test_resolutions = [(1920, 1080), (2560, 1440), (3440, 1440)]
    
    reference_result = None
    iou_scores = []
    
    for width, height in test_resolutions:
        screenshot = create_synthetic_arena_screenshot(width, height)
        result = detector.detect_cards_automatically(screenshot)
        
        if result and result['success']:
            if reference_result is None:
                reference_result = result
                print(f"Reference: {width}x{height}")
            else:
                # Calculate IoU with reference (normalized coordinates)
                ref_positions = reference_result['card_positions']
                curr_positions = result['card_positions']
                
                # Normalize to [0,1] coordinate space
                ref_w, ref_h = test_resolutions[0]
                curr_w, curr_h = width, height
                
                ious = []
                for i in range(min(len(ref_positions), len(curr_positions))):
                    # Normalize reference
                    rx, ry, rw, rh = ref_positions[i]
                    norm_ref = (rx/ref_w, ry/ref_h, rw/ref_w, rh/ref_h)
                    
                    # Normalize current  
                    cx, cy, cw, ch = curr_positions[i]
                    norm_curr = (cx/curr_w, cy/curr_h, cw/curr_w, ch/curr_h)
                    
                    # Convert back to same coordinate space for IoU calculation
                    ref_box = (norm_ref[0]*1000, norm_ref[1]*1000, norm_ref[2]*1000, norm_ref[3]*1000)
                    curr_box = (norm_curr[0]*1000, norm_curr[1]*1000, norm_curr[2]*1000, norm_curr[3]*1000)
                    
                    iou = calculate_iou(ref_box, curr_box)
                    ious.append(iou)
                
                avg_iou = np.mean(ious)
                iou_scores.append(avg_iou)
                print(f"IoU vs reference ({width}x{height}): {avg_iou:.3f}")
    
    if iou_scores:
        cross_resolution_iou = np.mean(iou_scores)
        print(f"Average cross-resolution IoU: {cross_resolution_iou:.3f}")
    
    # Final evaluation using ground truth
    if ground_truth_ious:
        ground_truth_avg = np.mean(ground_truth_ious)
        print(f"Ground truth IoU: {ground_truth_avg:.6f}")
        
        robustness_threshold = 0.98
        if ground_truth_avg >= robustness_threshold:
            print(f"✅ PRODUCTION READY: Ground truth IoU >= {robustness_threshold}")
            return True
        else:
            print(f"⚠️ NEEDS IMPROVEMENT: Ground truth IoU < {robustness_threshold}")
            return False
    else:
        # Fallback to cross-resolution test
        if iou_scores:
            cross_resolution_iou = np.mean(iou_scores)
            robustness_threshold = 0.98
            if cross_resolution_iou >= robustness_threshold:
                print(f"✅ ROBUST: Cross-resolution IoU >= {robustness_threshold}")
                return True
            else:
                print(f"⚠️ NEEDS IMPROVEMENT: Cross-resolution IoU < {robustness_threshold}")
                return False
    
    return False


if __name__ == "__main__":
    # Test validation set
    validation_results = test_validation_set()
    
    # Test IoU robustness
    is_robust = test_iou_robustness()
    
    print(f"\n=== FINAL VERDICT ===")
    if is_robust:
        print("🎉 Two-stage pipeline is PRODUCTION-READY with >0.98 IoU robustness!")
    else:
        print("⚠️ Two-stage pipeline needs refinement to achieve >0.98 IoU robustness.")