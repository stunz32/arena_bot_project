#!/usr/bin/env python3
"""
Metrics Logger for Arena Bot Detection System
Tracks detection performance, timing, and accuracy metrics in CSV format.
"""

import csv
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from debug_config import get_debug_config, is_debug_enabled

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Comprehensive metrics logging system for detection performance analysis.
    Tracks IoU, confidence, timing, and accuracy metrics with CSV export.
    """
    
    def __init__(self):
        self.config = get_debug_config()
        self.csv_file = self.config.METRICS['csv_file']
        self.performance_report_file = self.config.METRICS['performance_report']
        self.fields = self.config.METRICS['fields']
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
        
        # Performance tracking
        self.session_metrics = []
        self.method_performance = {}
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            try:
                with open(self.csv_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fields)
                    writer.writeheader()
                logger.info(f"Initialized metrics CSV: {self.csv_file}")
            except Exception as e:
                logger.error(f"Failed to initialize CSV file: {e}")
    
    def log_detection_metrics(self,
                            screenshot_file: str,
                            resolution: Tuple[int, int],
                            detection_method: str,
                            detected_boxes: List[Tuple[int, int, int, int]],
                            ground_truth_boxes: List[Tuple[int, int, int, int]] = None,
                            card_names: List[str] = None,
                            confidences: List[float] = None,
                            detection_time_ms: float = None,
                            anchor_score: float = None) -> Dict[str, Any]:
        """
        Log comprehensive detection metrics for a single screenshot analysis.
        
        Args:
            screenshot_file: Path to screenshot file
            resolution: (width, height) of screenshot
            detection_method: Name of detection method used
            detected_boxes: List of detected card regions
            ground_truth_boxes: List of ground truth regions (optional)
            card_names: Identified card names (optional)
            confidences: Detection confidence scores (optional)
            detection_time_ms: Detection timing in milliseconds
            anchor_score: Quality score for anchor detection (optional)
            
        Returns:
            Dictionary containing calculated metrics
        """
        if not is_debug_enabled():
            return {}
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate IoU metrics if ground truth available
        iou_metrics = self._calculate_iou_metrics(detected_boxes, ground_truth_boxes)
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(confidences)
        
        # Calculate detection accuracy
        accuracy_metrics = self._calculate_accuracy_metrics(detected_boxes, ground_truth_boxes)
        
        # Calculate box quality scores
        box_quality_score = self._calculate_box_quality_score(detected_boxes)
        
        # Calculate overall grade
        overall_grade = self._calculate_overall_grade(
            iou_metrics, confidence_metrics, accuracy_metrics, detection_time_ms
        )
        
        # Prepare metrics record
        metrics_record = {
            'timestamp': timestamp,
            'screenshot_file': screenshot_file,
            'resolution': f"{resolution[0]}x{resolution[1]}",
            'detection_method': detection_method,
            'card1_iou': iou_metrics.get('card1_iou', 0.0),
            'card2_iou': iou_metrics.get('card2_iou', 0.0),
            'card3_iou': iou_metrics.get('card3_iou', 0.0),
            'mean_iou': iou_metrics.get('mean_iou', 0.0),
            'card1_confidence': confidence_metrics.get('card1_confidence', 0.0),
            'card2_confidence': confidence_metrics.get('card2_confidence', 0.0),
            'card3_confidence': confidence_metrics.get('card3_confidence', 0.0),
            'mean_confidence': confidence_metrics.get('mean_confidence', 0.0),
            'detection_time_ms': detection_time_ms or 0.0,
            'total_cards_detected': len(detected_boxes),
            'miss_rate': accuracy_metrics.get('miss_rate', 0.0),
            'anchor_score': anchor_score or 0.0,
            'box_accuracy_score': box_quality_score,
            'overall_grade': overall_grade
        }
        
        # Log to CSV
        self._write_csv_record(metrics_record)
        
        # Update session tracking
        self.session_metrics.append(metrics_record)
        self._update_method_performance(detection_method, metrics_record)
        
        # Log summary to console
        self._log_metrics_summary(metrics_record)
        
        return metrics_record
    
    def _calculate_iou_metrics(self, detected_boxes: List[Tuple[int, int, int, int]],
                              ground_truth_boxes: List[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """Calculate IoU metrics for detected vs ground truth boxes."""
        if not ground_truth_boxes:
            return {'mean_iou': 0.0}
        
        # Find best matches between detected and ground truth
        iou_scores = []
        individual_ious = {}
        
        for i in range(min(len(detected_boxes), len(ground_truth_boxes))):
            if i < len(detected_boxes) and i < len(ground_truth_boxes):
                iou = self.config.calculate_iou(detected_boxes[i], ground_truth_boxes[i])
                iou_scores.append(iou)
                individual_ious[f'card{i+1}_iou'] = iou
        
        # Calculate mean IoU
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        individual_ious['mean_iou'] = mean_iou
        
        return individual_ious
    
    def _calculate_confidence_metrics(self, confidences: List[float] = None) -> Dict[str, float]:
        """Calculate confidence score metrics."""
        if not confidences:
            return {'mean_confidence': 0.0}
        
        # Individual confidence scores
        confidence_metrics = {}
        for i, conf in enumerate(confidences[:3]):  # Up to 3 cards
            confidence_metrics[f'card{i+1}_confidence'] = conf
        
        # Mean confidence
        confidence_metrics['mean_confidence'] = sum(confidences) / len(confidences)
        
        return confidence_metrics
    
    def _calculate_accuracy_metrics(self, detected_boxes: List[Tuple[int, int, int, int]],
                                   ground_truth_boxes: List[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """Calculate detection accuracy metrics."""
        if not ground_truth_boxes:
            return {'miss_rate': 0.0}
        
        expected_cards = len(ground_truth_boxes)
        detected_cards = len(detected_boxes)
        
        # Calculate miss rate
        miss_rate = max(0.0, (expected_cards - detected_cards) / expected_cards)
        
        return {
            'miss_rate': miss_rate,
            'detection_rate': 1.0 - miss_rate,
            'cards_expected': expected_cards,
            'cards_detected': detected_cards
        }
    
    def _calculate_box_quality_score(self, detected_boxes: List[Tuple[int, int, int, int]]) -> float:
        """Calculate overall quality score for detected boxes."""
        if not detected_boxes:
            return 0.0
        
        quality_scores = []
        for box in detected_boxes:
            validation = self.config.validate_box(box)
            quality_scores.append(validation['quality_score'])
        
        return sum(quality_scores) / len(quality_scores)
    
    def _calculate_overall_grade(self, iou_metrics: Dict, confidence_metrics: Dict,
                               accuracy_metrics: Dict, detection_time_ms: float = None) -> str:
        """Calculate overall grade (A-F) based on all metrics."""
        score = 0.0
        
        # IoU score (40% weight)
        mean_iou = iou_metrics.get('mean_iou', 0.0)
        iou_score = min(1.0, mean_iou / self.config.THRESHOLDS['min_iou'])
        score += iou_score * 0.4
        
        # Confidence score (30% weight)
        mean_confidence = confidence_metrics.get('mean_confidence', 0.0)
        conf_score = min(1.0, mean_confidence / self.config.THRESHOLDS['min_confidence'])
        score += conf_score * 0.3
        
        # Accuracy score (20% weight)
        miss_rate = accuracy_metrics.get('miss_rate', 1.0)
        accuracy_score = max(0.0, 1.0 - miss_rate / self.config.THRESHOLDS['max_miss_rate'])
        score += accuracy_score * 0.2
        
        # Timing score (10% weight)
        if detection_time_ms is not None:
            timing_score = max(0.0, 1.0 - detection_time_ms / self.config.THRESHOLDS['max_detection_time_ms'])
            score += timing_score * 0.1
        
        # Convert to letter grade
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _write_csv_record(self, record: Dict[str, Any]):
        """Write metrics record to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fields)
                writer.writerow(record)
        except Exception as e:
            logger.error(f"Failed to write CSV record: {e}")
    
    def _update_method_performance(self, method: str, record: Dict[str, Any]):
        """Update performance tracking for detection method."""
        if method not in self.method_performance:
            self.method_performance[method] = {
                'total_tests': 0,
                'total_iou': 0.0,
                'total_confidence': 0.0,
                'total_time': 0.0,
                'grades': []
            }
        
        perf = self.method_performance[method]
        perf['total_tests'] += 1
        perf['total_iou'] += record.get('mean_iou', 0.0)
        perf['total_confidence'] += record.get('mean_confidence', 0.0)
        perf['total_time'] += record.get('detection_time_ms', 0.0)
        perf['grades'].append(record.get('overall_grade', 'F'))
    
    def _log_metrics_summary(self, record: Dict[str, Any]):
        """Log metrics summary to console."""
        method = record['detection_method']
        mean_iou = record['mean_iou']
        mean_conf = record['mean_confidence']
        timing = record['detection_time_ms']
        grade = record['overall_grade']
        
        # Color coding for console output
        grade_colors = {'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🟠', 'F': '🔴'}
        grade_icon = grade_colors.get(grade, '⚪')
        
        logger.info(f"📊 METRICS [{method}] {grade_icon} Grade: {grade}")
        logger.info(f"   IoU: {mean_iou:.3f} | Confidence: {mean_conf:.3f} | Time: {timing:.1f}ms")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.session_metrics:
            return {'error': 'No metrics data available'}
        
        report = {
            'session_summary': {
                'total_tests': len(self.session_metrics),
                'timestamp': datetime.now().isoformat(),
                'average_iou': sum(m['mean_iou'] for m in self.session_metrics) / len(self.session_metrics),
                'average_confidence': sum(m['mean_confidence'] for m in self.session_metrics) / len(self.session_metrics),
                'average_timing': sum(m['detection_time_ms'] for m in self.session_metrics) / len(self.session_metrics),
            },
            'method_comparison': {},
            'grade_distribution': {},
            'performance_thresholds': self.config.THRESHOLDS
        }
        
        # Method comparison
        for method, perf in self.method_performance.items():
            if perf['total_tests'] > 0:
                report['method_comparison'][method] = {
                    'tests': perf['total_tests'],
                    'avg_iou': perf['total_iou'] / perf['total_tests'],
                    'avg_confidence': perf['total_confidence'] / perf['total_tests'],
                    'avg_time_ms': perf['total_time'] / perf['total_tests'],
                    'grade_distribution': {grade: perf['grades'].count(grade) for grade in 'ABCDF'}
                }
        
        # Overall grade distribution
        all_grades = [m['overall_grade'] for m in self.session_metrics]
        report['grade_distribution'] = {grade: all_grades.count(grade) for grade in 'ABCDF'}
        
        # Save report to file
        try:
            with open(self.performance_report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved: {self.performance_report_file}")
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")
        
        return report
    
    def get_method_ranking(self) -> List[Tuple[str, float]]:
        """Get ranking of detection methods by overall performance."""
        rankings = []
        
        for method, perf in self.method_performance.items():
            if perf['total_tests'] > 0:
                # Calculate composite score
                avg_iou = perf['total_iou'] / perf['total_tests']
                avg_conf = perf['total_confidence'] / perf['total_tests']
                avg_time = perf['total_time'] / perf['total_tests']
                
                # Score calculation (higher is better)
                composite_score = (
                    avg_iou * 0.4 +  # IoU weight
                    avg_conf * 0.3 +  # Confidence weight
                    (1.0 - min(1.0, avg_time / 100)) * 0.3  # Speed weight (inverted)
                )
                
                rankings.append((method, composite_score))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def check_performance_thresholds(self) -> Dict[str, bool]:
        """Check if current performance meets defined thresholds."""
        if not self.session_metrics:
            return {'error': 'No metrics available'}
        
        recent_metrics = self.session_metrics[-10:]  # Last 10 tests
        
        avg_iou = sum(m['mean_iou'] for m in recent_metrics) / len(recent_metrics)
        avg_conf = sum(m['mean_confidence'] for m in recent_metrics) / len(recent_metrics)
        avg_time = sum(m['detection_time_ms'] for m in recent_metrics) / len(recent_metrics)
        avg_miss_rate = sum(m['miss_rate'] for m in recent_metrics) / len(recent_metrics)
        
        return {
            'iou_threshold_met': avg_iou >= self.config.THRESHOLDS['min_iou'],
            'confidence_threshold_met': avg_conf >= self.config.THRESHOLDS['min_confidence'],
            'timing_threshold_met': avg_time <= self.config.THRESHOLDS['max_detection_time_ms'],
            'miss_rate_threshold_met': avg_miss_rate <= self.config.THRESHOLDS['max_miss_rate'],
            'current_metrics': {
                'avg_iou': avg_iou,
                'avg_confidence': avg_conf,
                'avg_timing_ms': avg_time,
                'avg_miss_rate': avg_miss_rate
            }
        }

# Global metrics logger instance
metrics_logger = MetricsLogger()

def log_detection_metrics(*args, **kwargs) -> Dict[str, Any]:
    """Convenience function for logging detection metrics."""
    return metrics_logger.log_detection_metrics(*args, **kwargs)

def generate_performance_report() -> Dict[str, Any]:
    """Convenience function for generating performance report."""
    return metrics_logger.generate_performance_report()

def check_performance_thresholds() -> Dict[str, bool]:
    """Convenience function for checking performance thresholds."""
    return metrics_logger.check_performance_thresholds()