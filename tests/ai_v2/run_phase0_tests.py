#!/usr/bin/env python3
"""
Phase 0 Comprehensive Test Runner
Executes all Phase 0 tests with detailed reporting and quality gates
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite(test_file: Path) -> Dict[str, Any]:
    """Run a specific test suite and return results"""
    print(f"\n🧪 Running {test_file.name}...")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output and coverage
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            str(test_file), 
            '-v', 
            '--tb=short',
            '--disable-warnings'
        ], 
        capture_output=True, 
        text=True,
        cwd=project_root
        )
        
        duration = time.time() - start_time
        
        return {
            'test_file': test_file.name,
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'test_file': test_file.name,
            'return_code': -1,
            'duration': duration,
            'stdout': '',
            'stderr': str(e),
            'success': False,
            'error': str(e)
        }

def parse_test_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """Parse pytest output to extract test statistics"""
    stdout = result['stdout']
    
    stats = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'warnings': 0
    }
    
    # Parse pytest summary line
    for line in stdout.split('\n'):
        if 'passed' in line and ('failed' in line or 'error' in line or 'warning' in line):
            # Parse summary line like "5 passed, 2 failed, 1 error in 0.45s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'passed' and i > 0:
                    stats['passed'] = int(parts[i-1])
                elif part == 'failed' and i > 0:
                    stats['failed'] = int(parts[i-1])
                elif part == 'error' and i > 0:
                    stats['errors'] = int(parts[i-1])
                elif part == 'skipped' and i > 0:
                    stats['skipped'] = int(parts[i-1])
                elif 'warning' in part and i > 0:
                    stats['warnings'] = int(parts[i-1])
                    
        elif line.strip().endswith('passed'):
            # Parse simple line like "5 passed in 0.45s"
            parts = line.split()
            if len(parts) >= 2 and parts[1] == 'passed':
                stats['passed'] = int(parts[0])
    
    stats['total_tests'] = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
    
    return stats

def generate_test_report(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    total_duration = sum(r['duration'] for r in all_results)
    successful_suites = sum(1 for r in all_results if r['success'])
    
    # Aggregate test statistics
    total_stats = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'warnings': 0
    }
    
    suite_details = []
    
    for result in all_results:
        stats = parse_test_results(result)
        
        # Add to totals
        for key in total_stats:
            total_stats[key] += stats[key]
            
        suite_details.append({
            'name': result['test_file'],
            'success': result['success'],
            'duration': round(result['duration'], 3),
            'stats': stats,
            'return_code': result['return_code']
        })
    
    # Calculate quality metrics
    pass_rate = (total_stats['passed'] / max(1, total_stats['total_tests'])) * 100
    suite_success_rate = (successful_suites / len(all_results)) * 100
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase': 'Phase 0 - Foundation & Risk Mitigation',
        'summary': {
            'total_test_suites': len(all_results),
            'successful_suites': successful_suites,
            'suite_success_rate': round(suite_success_rate, 1),
            'total_duration_seconds': round(total_duration, 3),
            'total_tests': total_stats['total_tests'],
            'passed_tests': total_stats['passed'],
            'failed_tests': total_stats['failed'],
            'error_tests': total_stats['errors'],
            'skipped_tests': total_stats['skipped'],
            'test_pass_rate': round(pass_rate, 1)
        },
        'quality_gates': {
            'min_pass_rate': 90.0,
            'min_suite_success_rate': 80.0,
            'max_duration_seconds': 300.0,
            'pass_rate_met': pass_rate >= 90.0,
            'suite_success_rate_met': suite_success_rate >= 80.0,
            'duration_met': total_duration <= 300.0,
            'overall_quality_gate': (
                pass_rate >= 90.0 and 
                suite_success_rate >= 80.0 and 
                total_duration <= 300.0
            )
        },
        'suite_details': suite_details
    }
    
    return report

def print_test_report(report: Dict[str, Any]):
    """Print formatted test report"""
    print("\n" + "="*80)
    print(f"🎯 PHASE 0 TEST RESULTS - {report['timestamp']}")
    print("="*80)
    
    summary = report['summary']
    gates = report['quality_gates']
    
    print(f"\n📊 SUMMARY:")
    print(f"   Test Suites: {summary['successful_suites']}/{summary['total_test_suites']} successful ({summary['suite_success_rate']}%)")
    print(f"   Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['test_pass_rate']}%)")
    print(f"   Total Duration: {summary['total_duration_seconds']}s")
    
    if summary['failed_tests'] > 0:
        print(f"   ❌ Failed Tests: {summary['failed_tests']}")
    if summary['error_tests'] > 0:
        print(f"   🚨 Error Tests: {summary['error_tests']}")
    if summary['skipped_tests'] > 0:
        print(f"   ⏭️  Skipped Tests: {summary['skipped_tests']}")
        
    print(f"\n🚦 QUALITY GATES:")
    gate_status = "✅ PASSED" if gates['overall_quality_gate'] else "❌ FAILED"
    print(f"   Overall Status: {gate_status}")
    
    print(f"   Test Pass Rate: {summary['test_pass_rate']}% ({'✅' if gates['pass_rate_met'] else '❌'} ≥{gates['min_pass_rate']}%)")
    print(f"   Suite Success Rate: {summary['suite_success_rate']}% ({'✅' if gates['suite_success_rate_met'] else '❌'} ≥{gates['min_suite_success_rate']}%)")
    print(f"   Duration: {summary['total_duration_seconds']}s ({'✅' if gates['duration_met'] else '❌'} ≤{gates['max_duration_seconds']}s)")
    
    print(f"\n📋 SUITE DETAILS:")
    for suite in report['suite_details']:
        status = "✅" if suite['success'] else "❌"
        print(f"   {status} {suite['name']:<35} | {suite['stats']['passed']:>3} passed | {suite['duration']:>6.2f}s")
        
        if not suite['success']:
            print(f"      Failed: {suite['stats']['failed']}, Errors: {suite['stats']['errors']}")

def save_report(report: Dict[str, Any], output_file: Path):
    """Save report to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Full report saved to: {output_file}")

def show_failed_test_details(all_results: List[Dict[str, Any]]):
    """Show details for failed tests"""
    failed_results = [r for r in all_results if not r['success']]
    
    if not failed_results:
        return
        
    print(f"\n🔍 FAILED TEST DETAILS:")
    print("="*80)
    
    for result in failed_results:
        print(f"\n❌ {result['test_file']} (exit code: {result['return_code']})")
        print("-" * 60)
        
        if result['stderr']:
            print("STDERR:")
            print(result['stderr'][:1000])  # Limit output
            if len(result['stderr']) > 1000:
                print("... (truncated)")
                
        if result['stdout']:
            # Show only the failure summary from stdout
            lines = result['stdout'].split('\n')
            in_failure_section = False
            failure_lines = []
            
            for line in lines:
                if 'FAILURES' in line or 'ERRORS' in line:
                    in_failure_section = True
                    failure_lines.append(line)
                elif in_failure_section and line.startswith('='):
                    break
                elif in_failure_section:
                    failure_lines.append(line)
                    
            if failure_lines:
                print("FAILURES:")
                print('\n'.join(failure_lines[:50]))  # Limit lines
                if len(failure_lines) > 50:
                    print("... (truncated)")

def main():
    """Main test runner"""
    print("🚀 Starting Phase 0 Comprehensive Test Suite")
    print("="*80)
    
    # Find all test files
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "test_data_models.py",
        test_dir / "test_config_manager.py", 
        test_dir / "test_monitoring.py",
        test_dir / "test_dependency_validation.py"
    ]
    
    # Check that all test files exist
    missing_files = [f for f in test_files if not f.exists()]
    if missing_files:
        print(f"❌ Missing test files: {[f.name for f in missing_files]}")
        return 1
        
    print(f"📋 Found {len(test_files)} test suites to run:")
    for test_file in test_files:
        print(f"   • {test_file.name}")
    
    # Run all test suites
    all_results = []
    
    for test_file in test_files:
        result = run_test_suite(test_file)
        all_results.append(result)
        
        # Print immediate result
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"   {status} - {result['duration']:.2f}s")
        
        if not result['success']:
            print(f"      Return code: {result['return_code']}")
    
    # Generate and display report
    report = generate_test_report(all_results)
    print_test_report(report)
    
    # Save detailed report
    report_file = project_root / "phase0_test_report.json"
    save_report(report, report_file)
    
    # Show failed test details if any
    show_failed_test_details(all_results)
    
    # Final recommendations
    if report['quality_gates']['overall_quality_gate']:
        print(f"\n🎉 Phase 0 testing completed successfully!")
        print(f"   All quality gates passed. Ready to proceed to Phase 1.")
        return 0
    else:
        print(f"\n⚠️  Phase 0 testing completed with issues.")
        print(f"   Please address failing tests before proceeding to Phase 1.")
        
        # Specific recommendations
        gates = report['quality_gates']
        if not gates['pass_rate_met']:
            print(f"   • Improve test pass rate (currently {report['summary']['test_pass_rate']}%)")
        if not gates['suite_success_rate_met']:
            print(f"   • Fix failing test suites (currently {report['summary']['suite_success_rate']}% successful)")
        if not gates['duration_met']:
            print(f"   • Optimize test performance (currently {report['summary']['total_duration_seconds']}s)")
            
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)