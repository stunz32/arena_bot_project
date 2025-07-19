#!/usr/bin/env python3
"""
Intelligent Debug System - Quick Start Script
Demonstrates the complete debugging and validation pipeline for Arena Bot detection.
"""

import sys
import os
import time
from pathlib import Path

# Add project modules
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("🎯 ARENA BOT INTELLIGENT DEBUG SYSTEM")
    print("=" * 80)
    print("Advanced computer vision debugging with visual validation")
    print("=" * 80)
    
    try:
        # Import debug modules
        from debug_config import enable_debug, get_debug_config
        from validation_suite import run_full_validation, check_system_health
        from calibration_system import diagnose_detection_issues, run_automatic_calibration
        from metrics_logger import generate_performance_report
        
        # Enable debug mode
        enable_debug()
        print("🐛 Debug mode enabled - Visual debugging active")
        
        # Check system health first
        print("\n🏥 SYSTEM HEALTH CHECK:")
        print("-" * 40)
        health_ok = check_system_health()
        if health_ok:
            print("✅ System health: OK")
        else:
            print("❌ System health: FAILED")
            print("   Please check your installation and try again")
            return 1
        
        # Show user options
        print("\n🎮 DEBUGGING OPTIONS:")
        print("-" * 40)
        print("1. Run Full Validation Suite (recommended)")
        print("2. Diagnose Detection Issues")
        print("3. Run Automatic Calibration")
        print("4. Test GUI with Debug Mode")
        print("5. Show Performance Report")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\n👉 Select option (1-6): ").strip()
                
                if choice == "1":
                    print("\n🚀 RUNNING FULL VALIDATION SUITE...")
                    print("-" * 50)
                    results = run_full_validation()
                    
                    # Show summary
                    overall = results.get('overall_scores', {})
                    print(f"\n📊 VALIDATION RESULTS:")
                    print(f"   🏆 Best Method: {overall.get('best_method', 'Unknown')}")
                    print(f"   📈 Average IoU: {overall.get('average_iou', 0):.3f}")
                    print(f"   ⏱️ Average Time: {overall.get('average_timing', 0):.1f}ms")
                    print(f"   ✅ Pass Rate: {overall.get('overall_pass_rate', 0):.1%}")
                    
                    # Show debug images location
                    debug_config = get_debug_config()
                    print(f"\n🖼️ Debug images saved to: {debug_config.debug_frames_dir}")
                    print(f"📊 Metrics saved to: {debug_config.METRICS['csv_file']}")
                
                elif choice == "2":
                    print("\n🔍 DIAGNOSING DETECTION ISSUES...")
                    print("-" * 50)
                    diagnosis = diagnose_detection_issues()
                    
                    print(f"🎯 Issues Found: {len(diagnosis['issues_found'])}")
                    for issue in diagnosis['issues_found']:
                        print(f"   ⚠️ {issue}")
                    
                    print(f"\n💡 Recommendations:")
                    for rec in diagnosis['recommendations']:
                        print(f"   📝 {rec}")
                    
                    print(f"\n🔥 Severity: {diagnosis['severity'].upper()}")
                
                elif choice == "3":
                    print("\n🔧 RUNNING AUTOMATIC CALIBRATION...")
                    print("-" * 50)
                    print("This may take a few minutes...")
                    
                    cal_results = run_automatic_calibration()
                    improvement = cal_results.get('improvement', 0)
                    
                    if improvement > 0:
                        print(f"✅ Calibration successful!")
                        print(f"   📈 Performance improved by {improvement:.3f}")
                        print(f"   🎯 New score: {cal_results['final_score']:.3f}")
                    else:
                        print("📊 No improvement found - current parameters are optimal")
                
                elif choice == "4":
                    print("\n🎮 LAUNCHING GUI WITH DEBUG MODE...")
                    print("-" * 50)
                    print("Instructions:")
                    print("1. The GUI will launch with debug mode enabled")
                    print("2. Check the 🐛 DEBUG checkbox in the interface")
                    print("3. Use 'Simple Working' detection method")
                    print("4. Click '📸 ANALYZE SCREENSHOT' to test")
                    print("5. Check debug_frames/ folder for annotated images")
                    print("6. Click '📊 REPORT' to view performance metrics")
                    
                    # Launch GUI
                    try:
                        from integrated_arena_bot_gui import IntegratedArenaBotGUI
                        import tkinter as tk
                        
                        root = tk.Tk()
                        app = IntegratedArenaBotGUI()
                        
                        # Auto-enable debug mode
                        if hasattr(app, 'debug_enabled'):
                            app.debug_enabled.set(True)
                            app.toggle_debug_mode()
                        
                        print("✅ GUI launched with debug mode enabled")
                        root.mainloop()
                        
                    except Exception as e:
                        print(f"❌ Failed to launch GUI: {e}")
                        print("   Try running: python integrated_arena_bot_gui.py")
                
                elif choice == "5":
                    print("\n📊 GENERATING PERFORMANCE REPORT...")
                    print("-" * 50)
                    try:
                        report = generate_performance_report()
                        
                        if 'error' in report:
                            print(f"❌ {report['error']}")
                            print("   Run validation tests first to generate data")
                        else:
                            summary = report.get('session_summary', {})
                            print(f"📈 Total Tests: {summary.get('total_tests', 0)}")
                            print(f"📊 Average IoU: {summary.get('average_iou', 0):.3f}")
                            print(f"⏱️ Average Time: {summary.get('average_timing', 0):.1f}ms")
                            
                            # Show method comparison
                            methods = report.get('method_comparison', {})
                            if methods:
                                print(f"\n🔬 METHOD PERFORMANCE:")
                                for method, stats in methods.items():
                                    print(f"   {method}: IoU={stats['avg_iou']:.3f}, "
                                          f"Time={stats['avg_time_ms']:.1f}ms")
                    except Exception as e:
                        print(f"❌ Failed to generate report: {e}")
                
                elif choice == "6":
                    print("\n👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    continue
                
                # Ask if user wants to continue
                if choice in ["1", "2", "3", "5"]:
                    cont = input("\n🔄 Continue with another option? (y/n): ").strip().lower()
                    if cont not in ['y', 'yes']:
                        print("\n👋 Goodbye!")
                        break
                        
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure all dependencies are installed")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def show_help():
    """Show help information."""
    print("🎯 ARENA BOT INTELLIGENT DEBUG SYSTEM")
    print("=" * 60)
    print("\nThis system provides advanced debugging capabilities:")
    print("\n📊 FEATURES:")
    print("  • Visual debug overlays with IoU validation")
    print("  • Automated performance testing and metrics")
    print("  • Intelligent parameter calibration")
    print("  • Cross-resolution compatibility testing")
    print("  • Real-time detection quality assessment")
    print("\n🎮 USAGE:")
    print("  python run_intelligent_debug.py     # Interactive mode")
    print("  python run_intelligent_debug.py --help  # Show this help")
    print("\n🔧 DEBUG FILES:")
    print("  debug_frames/     # Annotated debug images")
    print("  debug_data/       # Metrics and performance data")
    print("  validation_results.json  # Comprehensive test results")
    print("\n💡 TIP: Enable debug mode in GUI for real-time visualization!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_help()
    else:
        sys.exit(main())