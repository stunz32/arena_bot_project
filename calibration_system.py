#!/usr/bin/env python3
"""
Intelligent Calibration System for Arena Bot Detection
Automatic parameter tuning and optimization based on performance feedback.
"""

import json
import numpy as np
import cv2
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sys

# Add project modules
sys.path.insert(0, str(Path(__file__).parent))

from debug_config import get_debug_config
from validation_suite import ValidationSuite
from metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)

class CalibrationSystem:
    """
    Intelligent calibration system that automatically adjusts detection parameters
    based on performance feedback and validation results.
    """
    
    def __init__(self):
        self.config = get_debug_config()
        self.validation_suite = ValidationSuite()
        self.metrics_logger = MetricsLogger()
        
        # Calibration parameters
        self.parameter_ranges = {
            'coordinate_offsets': {
                'x_offset': (-50, 50, 5),      # (min, max, step)
                'y_offset': (-50, 50, 5),
                'width_scale': (0.8, 1.2, 0.05),
                'height_scale': (0.8, 1.2, 0.05)
            },
            'detection_thresholds': {
                'confidence_threshold': (0.5, 0.95, 0.05),
                'iou_threshold': (0.7, 0.95, 0.05),
                'timing_threshold': (50, 200, 10)
            },
            'image_processing': {
                'blur_kernel': (1, 9, 2),
                'contrast_alpha': (0.8, 1.5, 0.1),
                'brightness_beta': (-20, 20, 5)
            }
        }
        
        # Calibration history
        self.calibration_history = []
        self.best_parameters = {}
        self.current_parameters = self.load_current_parameters()
    
    def load_current_parameters(self) -> Dict[str, Any]:
        """Load current calibration parameters."""
        try:
            cal_file = self.config.debug_data_dir / 'calibration_parameters.json'
            if cal_file.exists():
                with open(cal_file, 'r') as f:
                    return json.load(f)
            else:
                return self.get_default_parameters()
        except Exception as e:
            logger.error(f"Failed to load calibration parameters: {e}")
            return self.get_default_parameters()
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default calibration parameters."""
        return {
            'coordinate_offsets': {
                'x_offset': 0,
                'y_offset': 0,
                'width_scale': 1.0,
                'height_scale': 1.0
            },
            'detection_thresholds': {
                'confidence_threshold': 0.8,
                'iou_threshold': 0.92,
                'timing_threshold': 100
            },
            'image_processing': {
                'blur_kernel': 3,
                'contrast_alpha': 1.0,
                'brightness_beta': 0
            }
        }
    
    def save_parameters(self, parameters: Dict[str, Any]):
        """Save calibration parameters to file."""
        try:
            cal_file = self.config.debug_data_dir / 'calibration_parameters.json'
            with open(cal_file, 'w') as f:
                json.dump(parameters, f, indent=2)
            logger.info(f"Calibration parameters saved: {cal_file}")
        except Exception as e:
            logger.error(f"Failed to save calibration parameters: {e}")
    
    def run_automatic_calibration(self, target_method: str = "simple_working") -> Dict[str, Any]:
        """
        Run automatic calibration to optimize detection parameters.
        
        Args:
            target_method: Detection method to optimize
            
        Returns:
            Calibration results with optimized parameters
        """
        logger.info(f"🔧 Starting automatic calibration for {target_method}")
        
        # Baseline performance test
        baseline_results = self.test_current_parameters(target_method)
        baseline_score = self.calculate_performance_score(baseline_results)
        
        logger.info(f"📊 Baseline performance score: {baseline_score:.3f}")
        
        # Parameter optimization
        best_score = baseline_score
        best_params = self.current_parameters.copy()
        
        optimization_results = {
            'baseline_score': baseline_score,
            'baseline_parameters': self.current_parameters.copy(),
            'optimization_attempts': [],
            'final_score': baseline_score,
            'final_parameters': best_params,
            'improvement': 0.0
        }
        
        # Grid search optimization
        for param_category, param_dict in self.parameter_ranges.items():
            logger.info(f"🔍 Optimizing {param_category} parameters...")
            
            for param_name, (min_val, max_val, step) in param_dict.items():
                logger.info(f"   Testing {param_name}...")
                
                # Test different values for this parameter
                test_values = np.arange(min_val, max_val + step, step)
                
                for test_value in test_values:
                    # Create test parameters
                    test_params = best_params.copy()
                    test_params[param_category][param_name] = test_value
                    
                    # Test performance with these parameters
                    test_results = self.test_parameters(target_method, test_params)
                    test_score = self.calculate_performance_score(test_results)
                    
                    # Record attempt
                    attempt = {
                        'parameter': f"{param_category}.{param_name}",
                        'value': test_value,
                        'score': test_score,
                        'improvement': test_score - baseline_score
                    }
                    optimization_results['optimization_attempts'].append(attempt)
                    
                    # Check if this is better
                    if test_score > best_score:
                        best_score = test_score
                        best_params = test_params.copy()
                        logger.info(f"      ✅ New best: {param_name}={test_value}, score={test_score:.3f}")
                    else:
                        logger.debug(f"      📊 {param_name}={test_value}, score={test_score:.3f}")
        
        # Update final results
        optimization_results['final_score'] = best_score
        optimization_results['final_parameters'] = best_params
        optimization_results['improvement'] = best_score - baseline_score
        
        # Save optimized parameters if improvement found
        if best_score > baseline_score:
            self.save_parameters(best_params)
            self.current_parameters = best_params
            logger.info(f"🎯 Calibration improved performance by {optimization_results['improvement']:.3f}")
        else:
            logger.info("📊 No improvement found, keeping current parameters")
        
        # Save calibration history
        self.calibration_history.append(optimization_results)
        self.save_calibration_history()
        
        return optimization_results
    
    def test_current_parameters(self, method: str) -> Dict[str, Any]:
        """Test current parameters with validation suite."""
        return self.validation_suite.test_detection_method(method, save_debug=False)
    
    def test_parameters(self, method: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Test specific parameters by temporarily applying them."""
        # Save current parameters
        original_params = self.current_parameters.copy()
        
        try:
            # Apply test parameters
            self.current_parameters = parameters
            
            # Run validation test
            results = self.test_current_parameters(method)
            
            return results
            
        finally:
            # Restore original parameters
            self.current_parameters = original_params
    
    def calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite performance score from validation results."""
        if not results or results.get('tests_run', 0) == 0:
            return 0.0
        
        # Extract metrics
        avg_iou = results.get('avg_iou', 0.0)
        avg_timing = results.get('avg_timing', 999.0)
        pass_rate = results.get('pass_rate', 0.0)
        
        # Normalize timing score (lower is better)
        timing_score = max(0.0, 1.0 - avg_timing / 200.0)  # Normalize to 200ms max
        
        # Composite score (weighted average)
        score = (
            avg_iou * 0.4 +           # IoU accuracy (40%)
            pass_rate * 0.4 +         # Pass rate (40%)
            timing_score * 0.2        # Speed (20%)
        )
        
        return score
    
    def diagnose_detection_issues(self, method: str = "simple_working") -> Dict[str, Any]:
        """
        Diagnose common detection issues and suggest fixes.
        
        Returns:
            Dictionary with diagnosed issues and recommended fixes
        """
        logger.info(f"🔍 Diagnosing detection issues for {method}")
        
        # Run comprehensive validation
        validation_results = self.validation_suite.test_detection_method(method, save_debug=True)
        
        diagnosis = {
            'issues_found': [],
            'recommendations': [],
            'severity': 'low',  # low, medium, high, critical
            'validation_results': validation_results
        }
        
        if validation_results['tests_run'] == 0:
            diagnosis['issues_found'].append("No tests could be run - check ground truth data")
            diagnosis['severity'] = 'critical'
            return diagnosis
        
        avg_iou = validation_results.get('avg_iou', 0.0)
        avg_timing = validation_results.get('avg_timing', 0.0)
        pass_rate = validation_results.get('pass_rate', 0.0)
        
        # Analyze IoU issues
        if avg_iou < 0.7:
            diagnosis['issues_found'].append(f"Low IoU accuracy: {avg_iou:.3f} (target: 0.92+)")
            diagnosis['recommendations'].extend([
                "Check coordinate scaling for current resolution",
                "Verify anchor point detection accuracy",
                "Consider adjusting detection offsets",
                "Review ground truth data accuracy"
            ])
            diagnosis['severity'] = 'high'
        
        elif avg_iou < 0.9:
            diagnosis['issues_found'].append(f"Moderate IoU accuracy: {avg_iou:.3f} (target: 0.92+)")
            diagnosis['recommendations'].extend([
                "Fine-tune coordinate offsets",
                "Consider template matching improvements"
            ])
            if diagnosis['severity'] == 'low':
                diagnosis['severity'] = 'medium'
        
        # Analyze timing issues
        if avg_timing > 150:
            diagnosis['issues_found'].append(f"Slow detection: {avg_timing:.1f}ms (target: <100ms)")
            diagnosis['recommendations'].extend([
                "Enable region optimization",
                "Reduce image preprocessing steps",
                "Check for memory leaks",
                "Use lighter detection algorithms"
            ])
            if diagnosis['severity'] in ['low', 'medium']:
                diagnosis['severity'] = 'medium'
        
        # Analyze pass rate issues
        if pass_rate < 0.8:
            diagnosis['issues_found'].append(f"Low pass rate: {pass_rate:.1%} (target: 80%+)")
            diagnosis['recommendations'].extend([
                "Run automatic calibration",
                "Check detection thresholds",
                "Verify interface detection is working"
            ])
            diagnosis['severity'] = 'high'
        
        # Check grade distribution
        grade_dist = validation_results.get('grade_distribution', {})
        if grade_dist.get('F', 0) > grade_dist.get('A', 0):
            diagnosis['issues_found'].append("High failure rate in detection quality")
            diagnosis['recommendations'].append("Consider algorithm improvements or parameter tuning")
            diagnosis['severity'] = 'high'
        
        # Suggest automatic calibration if issues found
        if diagnosis['issues_found']:
            diagnosis['recommendations'].append("Run automatic calibration to optimize parameters")
        
        return diagnosis
    
    def generate_calibration_report(self) -> str:
        """Generate human-readable calibration report."""
        if not self.calibration_history:
            return "No calibration history available."
        
        latest = self.calibration_history[-1]
        
        report = "🔧 CALIBRATION SYSTEM REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Current status
        report += f"📊 Current Performance Score: {latest['final_score']:.3f}\n"
        report += f"📈 Improvement from Baseline: {latest['improvement']:.3f}\n"
        report += f"🔄 Total Calibration Runs: {len(self.calibration_history)}\n\n"
        
        # Best improvements
        if latest['optimization_attempts']:
            best_attempts = sorted(
                latest['optimization_attempts'], 
                key=lambda x: x['improvement'], 
                reverse=True
            )[:5]
            
            report += "🎯 TOP PARAMETER IMPROVEMENTS:\n"
            for attempt in best_attempts:
                if attempt['improvement'] > 0:
                    report += f"   {attempt['parameter']}: {attempt['value']} "
                    report += f"(+{attempt['improvement']:.3f})\n"
        
        # Current parameters
        report += "\n⚙️ CURRENT PARAMETERS:\n"
        for category, params in self.current_parameters.items():
            report += f"   {category}:\n"
            for param, value in params.items():
                report += f"      {param}: {value}\n"
        
        return report
    
    def save_calibration_history(self):
        """Save calibration history to file."""
        try:
            history_file = self.config.debug_data_dir / 'calibration_history.json'
            with open(history_file, 'w') as f:
                json.dump(self.calibration_history, f, indent=2, default=str)
            logger.info(f"Calibration history saved: {history_file}")
        except Exception as e:
            logger.error(f"Failed to save calibration history: {e}")
    
    def suggest_parameter_adjustments(self, issues: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest specific parameter adjustments based on diagnosed issues."""
        suggestions = {
            'coordinate_adjustments': {},
            'threshold_adjustments': {},
            'processing_adjustments': {}
        }
        
        # Analyze issues and suggest fixes
        for issue in issues.get('issues_found', []):
            if "Low IoU" in issue:
                suggestions['coordinate_adjustments'] = {
                    'x_offset': [-10, -5, 0, 5, 10],
                    'y_offset': [-10, -5, 0, 5, 10],
                    'width_scale': [0.95, 1.0, 1.05],
                    'height_scale': [0.95, 1.0, 1.05]
                }
            
            if "Slow detection" in issue:
                suggestions['processing_adjustments'] = {
                    'blur_kernel': [1, 3],  # Reduce blur
                    'enable_optimization': True
                }
            
            if "Low pass rate" in issue:
                suggestions['threshold_adjustments'] = {
                    'confidence_threshold': [0.6, 0.7, 0.8],
                    'iou_threshold': [0.85, 0.90, 0.92]
                }
        
        return suggestions


# Convenience functions
def run_automatic_calibration(method: str = "simple_working") -> Dict[str, Any]:
    """Run automatic calibration - convenience function."""
    calibrator = CalibrationSystem()
    return calibrator.run_automatic_calibration(method)

def diagnose_detection_issues(method: str = "simple_working") -> Dict[str, Any]:
    """Diagnose detection issues - convenience function."""
    calibrator = CalibrationSystem()
    return calibrator.diagnose_detection_issues(method)

def get_calibration_report() -> str:
    """Get calibration report - convenience function."""
    calibrator = CalibrationSystem()
    return calibrator.generate_calibration_report()


if __name__ == "__main__":
    # CLI interface for calibration system
    import argparse
    
    parser = argparse.ArgumentParser(description="Arena Bot Calibration System")
    parser.add_argument("--calibrate", help="Run automatic calibration for method")
    parser.add_argument("--diagnose", help="Diagnose issues for method")
    parser.add_argument("--report", action="store_true", help="Show calibration report")
    
    args = parser.parse_args()
    
    if args.calibrate:
        print(f"Running automatic calibration for {args.calibrate}...")
        results = run_automatic_calibration(args.calibrate)
        print(f"Calibration complete. Improvement: {results['improvement']:.3f}")
    
    elif args.diagnose:
        print(f"Diagnosing detection issues for {args.diagnose}...")
        diagnosis = diagnose_detection_issues(args.diagnose)
        print(f"Issues found: {len(diagnosis['issues_found'])}")
        for issue in diagnosis['issues_found']:
            print(f"  - {issue}")
        print("\nRecommendations:")
        for rec in diagnosis['recommendations']:
            print(f"  - {rec}")
    
    elif args.report:
        print(get_calibration_report())
    
    else:
        print("Use --calibrate, --diagnose, or --report")
        parser.print_help()