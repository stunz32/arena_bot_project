"""
CLI Interface for Arena Bot

Provides command-line interface functions for replay mode, offline processing,
and diagnostic operations. Separates CLI logic from main entry point.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import glob

from .utils.debug_dump import begin_run, dump_image, dump_json, end_run, is_debug_active

logger = logging.getLogger(__name__)

# Mock timing data for offline mode
MOCK_STAGE_TIMINGS = {
    "coordinates": 45.2,
    "eligibility_filter": 18.7,
    "histogram_match": 124.3,
    "template_validation": 67.8,
    "ai_advisor": 142.1,
    "ui_render": 28.4
}


def load_sidecar_labels(image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load sidecar JSON labels for an image if they exist.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Sidecar data dict or None if no sidecar exists
    """
    sidecar_path = image_path.with_suffix('.labels.json')
    
    if not sidecar_path.exists():
        return None
    
    try:
        with open(sidecar_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load sidecar {sidecar_path}: {e}")
        return None


def process_frame_offline(image_path: Path, sidecar_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Process a single frame in offline mode using sidecar data or mock detection.
    
    Args:
        image_path: Path to the image file
        sidecar_data: Optional sidecar labels data
        
    Returns:
        Frame processing result with cards, scores, and timings
    """
    start_time = time.time()
    
    # Use sidecar data if available, otherwise create mock detection
    if sidecar_data and "cards" in sidecar_data:
        cards = sidecar_data["cards"]
        logger.debug(f"Using sidecar data for {image_path.name}: {len(cards)} cards")
    else:
        # Mock detection fallback
        cards = [
            {"id": "mock_card_1", "name": "Mock Fireball", "mana_cost": 4, "tier_score": 75.0},
            {"id": "mock_card_2", "name": "Mock Frostbolt", "mana_cost": 2, "tier_score": 82.0},
            {"id": "mock_card_3", "name": "Mock Flamestrike", "mana_cost": 7, "tier_score": 68.0}
        ]
        logger.debug(f"Using mock detection for {image_path.name}")
    
    # Sort cards by tier_score descending
    cards_sorted = sorted(cards, key=lambda x: x.get("tier_score", 0), reverse=True)
    
    # Calculate total processing time (mock for offline)
    total_time_ms = sum(MOCK_STAGE_TIMINGS.values())
    
    # Create result structure
    result = {
        "image_path": str(image_path),
        "timestamp": time.time(),
        "cards": cards_sorted,
        "processing_time_ms": total_time_ms,
        "stage_timings": MOCK_STAGE_TIMINGS.copy(),
        "offline_mode": True,
        "sidecar_used": sidecar_data is not None
    }
    
    # Dump debug artifacts if debug run is active
    if is_debug_active():
        # Dump input image
        dump_image(image_path, f"frame_{image_path.stem}")
        
        # Dump processing result
        dump_json(result, f"result_{image_path.stem}")
        
        # Dump stage timings
        dump_json({
            "stage_timings": MOCK_STAGE_TIMINGS,
            "total_ms": total_time_ms,
            "image": image_path.name
        }, f"timings_{image_path.stem}")
    
    return result


def run_replay(paths: str, offline: bool = True, debug_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Run replay mode on a directory or glob pattern of images.
    
    Args:
        paths: Directory path or glob pattern for images
        offline: Whether to use offline mode (sidecar labels)
        debug_tag: Optional debug tag for debug dump
        
    Returns:
        List of per-frame processing results
    """
    logger.info(f"Starting replay mode: paths={paths}, offline={offline}, debug_tag={debug_tag}")
    
    # Start debug run if requested
    if debug_tag:
        begin_run(debug_tag)
    
    try:
        # Resolve paths to image files
        path_obj = Path(paths)
        
        if path_obj.is_dir():
            # Directory - find all PNG files
            image_files = list(path_obj.glob("*.png"))
        elif "*" in paths or "?" in paths:
            # Glob pattern
            image_files = [Path(p) for p in glob.glob(paths) if p.endswith('.png')]
        else:
            # Single file
            image_files = [path_obj] if path_obj.exists() and path_obj.suffix == '.png' else []
        
        if not image_files:
            logger.warning(f"No PNG files found in: {paths}")
            return []
        
        # Sort files for consistent processing order
        image_files.sort()
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        results = []
        
        for image_path in image_files:
            logger.debug(f"Processing {image_path.name}")
            
            # Load sidecar data if in offline mode
            sidecar_data = None
            if offline:
                sidecar_data = load_sidecar_labels(image_path)
            
            # Process the frame
            result = process_frame_offline(image_path, sidecar_data)
            results.append(result)
            
            # Print concise summary line
            cards = result["cards"]
            top_card = cards[0] if cards else {"name": "None", "tier_score": 0}
            print(f"Frame {image_path.name}: {len(cards)} cards, top={top_card['name']} (score={top_card.get('tier_score', 0)})")
        
        logger.info(f"Replay completed: {len(results)} frames processed")
        return results
        
    finally:
        # End debug run if active
        if debug_tag:
            end_run()


def print_diagnostic_output(results: List[Dict[str, Any]]) -> None:
    """
    Print diagnostic timing output for replay results with budget validation.
    
    Args:
        results: List of frame processing results
    """
    from arena_bot.utils.timing_utils import PerformanceTracker, format_timing_output
    
    if not results:
        print("No results to display diagnostics for")
        return
    
    print("\n📊 Diagnostic Timing Summary:")
    print("=" * 50)
    
    # Calculate average timings across all frames
    stage_totals = {}
    frame_count = len(results)
    skipped_stages = set()
    
    for result in results:
        stage_timings = result.get("stage_timings", {})
        for stage, duration in stage_timings.items():
            stage_totals[stage] = stage_totals.get(stage, 0) + duration
            
        # Track skipped stages
        if result.get("sidecar_used", False):
            skipped_stages.update(["histogram_match", "template_validation", "ai_advisor"])
    
    # Create performance tracker for budget validation
    tracker = PerformanceTracker()
    tracker.start_session()
    
    # Record average timings
    for stage, total in stage_totals.items():
        avg_ms = total / frame_count
        is_skipped = stage in skipped_stages
        tracker.record_stage(stage, avg_ms, skipped=is_skipped)
    
    # Get budget analysis
    summary = tracker.get_summary()
    
    # Print per-stage averages with budget status
    for stage, total in stage_totals.items():
        avg_ms = total / frame_count
        budget_ms = tracker.budgets.get(stage)
        is_skipped = stage in skipped_stages
        
        if is_skipped:
            status_icon = "⏭️"
            budget_info = " (skipped)"
        elif budget_ms:
            percentage = (avg_ms / budget_ms) * 100
            if avg_ms > budget_ms:
                status_icon = "🚨"
                budget_info = f" (>{budget_ms:.0f}ms budget, {percentage:.0f}%)"
            elif percentage >= 80:
                status_icon = "⚠️"
                budget_info = f" ({percentage:.0f}% of budget)"
            else:
                status_icon = "✅"
                budget_info = f" ({percentage:.0f}% of budget)"
        else:
            status_icon = "❓"
            budget_info = " (no budget)"
        
        print(f"  {status_icon} {stage:18}: {avg_ms:6.1f}ms{budget_info}")
    
    # Print total timing with budget check
    total_avg = summary['total_ms']
    total_budget = tracker.budgets.get('total')
    
    if total_budget:
        total_percentage = (total_avg / total_budget) * 100
        if total_avg > total_budget:
            total_status = "🚨"
            total_info = f" (>{total_budget:.0f}ms budget, {total_percentage:.0f}%)"
        else:
            total_status = "✅"
            total_info = f" ({total_percentage:.0f}% of budget)"
    else:
        total_status = "❓"
        total_info = " (no budget)"
    
    print(f"  {total_status} {'TOTAL':18}: {total_avg:6.1f}ms{total_info}")
    print(f"\nProcessed {frame_count} frame(s)")
    
    # Show summary
    print(f"Summary: {summary['summary']}")


def run_replay_from_cli(args) -> None:
    """
    Run replay mode from CLI arguments.
    
    Args:
        args: Parsed command line arguments
    """
    if not args.replay:
        print("❌ --replay flag required for replay mode")
        return
    
    # Run replay
    results = run_replay(
        paths=args.replay,
        offline=args.offline,
        debug_tag=args.debug_tag
    )
    
    if not results:
        print("❌ No frames processed")
        return
    
    print(f"✅ Processed {len(results)} frames")
    
    # Print diagnostics if requested
    if args.diag:
        print_diagnostic_output(results)
    
    # Show debug directory if debug was active
    if args.debug_tag:
        debug_base = Path(".debug_runs")
        if debug_base.exists():
            # Find most recent debug run
            debug_runs = sorted(debug_base.glob("*"), key=lambda p: p.stat().st_mtime)
            if debug_runs:
                latest_run = debug_runs[-1]
                print(f"\n🐛 Debug artifacts: {latest_run}")
                artifacts = list(latest_run.glob("*"))
                for artifact in sorted(artifacts):
                    print(f"   - {artifact.name}")


def run_live_smoke_from_cli(args) -> None:
    """
    Run live smoke mode from CLI arguments.
    
    Performs a single capture-and-render pass with diagnostics and exits.
    
    Args:
        args: Parsed command line arguments
    """
    from arena_bot.utils.live_test_gate import LiveTestGate
    from arena_bot.capture.capture_backend import AdaptiveCaptureManager
    from arena_bot.utils.timing_utils import PerformanceTracker
    from arena_bot.utils.debug_dump import DebugDumpManager
    from arena_bot.utils.live_diagnostics import get_live_diagnostics
    import time
    
    print("🧪 Live Smoke Mode - Single Capture Test")
    print("=" * 50)
    
    # Check live test requirements
    can_run, reason = LiveTestGate.check_live_test_requirements()
    if not can_run:
        print(f"❌ Cannot run live smoke test: {reason}")
        print("💡 To enable live testing:")
        print("   1. Set ARENA_LIVE_TESTS=1")
        print("   2. Run on Windows with GUI desktop")
        print("   3. Launch Hearthstone in windowed/borderless mode")
        return
    
    print(f"✅ Live test environment ready: {reason}")
    
    # Start debug dump if requested
    debug_manager = None
    if args.debug_tag:
        debug_manager = DebugDumpManager()
        debug_manager.start_dump(args.debug_tag + "_live_smoke")
        print(f"🐛 Debug dump started: {args.debug_tag}_live_smoke")
    
    # Initialize performance tracker and diagnostics
    tracker = PerformanceTracker()
    diagnostics = get_live_diagnostics()
    total_start = time.time()
    
    try:
        # Stage 1: Initialize capture manager
        print("\n📱 Stage 1: Initializing capture manager...")
        stage_start = time.time()
        
        capture_manager = AdaptiveCaptureManager()
        backend = capture_manager.get_active_backend()
        
        # Connect diagnostics
        diagnostics.set_capture_manager(capture_manager)
        
        init_duration = (time.time() - stage_start) * 1000
        tracker.record_stage('capture_init', init_duration)
        
        print(f"   ✅ Backend selected: {backend.get_name()}")
        print(f"   ⏱️ Initialization: {init_duration:.1f}ms")
        
        # Stage 2: Find Hearthstone window
        print("\n🎯 Stage 2: Finding Hearthstone window...")
        stage_start = time.time()
        
        window = capture_manager.find_hearthstone_window()
        if not window:
            raise RuntimeError("Hearthstone window not found")
        
        find_duration = (time.time() - stage_start) * 1000
        tracker.record_stage('window_find', find_duration)
        
        print(f"   ✅ Found: '{window.title}'")
        print(f"   📐 Size: {window.width}x{window.height}")
        print(f"   📍 Position: ({window.x}, {window.y})")
        print(f"   ⏱️ Window detection: {find_duration:.1f}ms")
        
        # Stage 3: Capture frame
        print("\n📸 Stage 3: Capturing frame...")
        stage_start = time.time()
        
        frame = capture_manager.capture_rect(window.x, window.y, window.width, window.height)
        
        capture_duration = (time.time() - stage_start) * 1000
        tracker.record_stage('frame_capture', capture_duration)
        
        # Record capture operation in diagnostics
        diagnostics.record_capture_operation(frame, capture_duration)
        
        # Verify frame
        if frame is None or frame.image is None:
            raise RuntimeError("Frame capture failed")
        
        height, width = frame.image.shape[:2]
        print(f"   ✅ Captured: {width}x{height}")
        print(f"   🔧 Backend: {frame.backend_name}")
        print(f"   ⏱️ Capture time: {frame.capture_duration_ms:.1f}ms")
        print(f"   ⏱️ Total capture stage: {capture_duration:.1f}ms")
        
        # Stage 4: Save debug artifacts
        if debug_manager:
            print("\n💾 Stage 4: Saving debug artifacts...")
            stage_start = time.time()
            
            # Save captured frame
            frame_path = debug_manager.save_image(frame.image, "live_smoke_frame")
            
            # Save metadata
            smoke_metadata = {
                'window_info': {
                    'title': window.title,
                    'handle': window.handle,
                    'size': [window.width, window.height],
                    'position': [window.x, window.y]
                },
                'frame_info': {
                    'backend_name': frame.backend_name,
                    'capture_duration_ms': frame.capture_duration_ms,
                    'image_size': [width, height],
                    'dpi_scale': frame.dpi_scale
                },
                'performance_stats': tracker.get_performance_stats()
            }
            
            metadata_path = debug_manager.save_json(smoke_metadata, "live_smoke_metadata")
            
            save_duration = (time.time() - stage_start) * 1000
            tracker.record_stage('debug_save', save_duration)
            
            print(f"   ✅ Frame saved: {frame_path}")
            print(f"   ✅ Metadata saved: {metadata_path}")
            print(f"   ⏱️ Save time: {save_duration:.1f}ms")
        
        # Calculate total time
        total_duration = (time.time() - total_start) * 1000
        tracker.record_stage('total', total_duration)
        
        # Print final diagnostics
        print("\n📊 Live Smoke Test Results")
        print("=" * 30)
        
        if args.diag:
            # Print detailed diagnostics
            stage_stats = tracker.get_performance_stats()
            for stage, stats in stage_stats.items():
                if stats['count'] > 0:
                    avg_ms = stats['total_ms'] / stats['count']
                    budget_ms = tracker.budgets.get(stage)
                    
                    if budget_ms:
                        percentage = (avg_ms / budget_ms) * 100
                        status_icon = "🚨" if percentage >= 100 else "⚠️" if percentage >= 80 else "✅"
                        budget_info = f" ({percentage:.0f}% of {budget_ms:.0f}ms budget)"
                    else:
                        status_icon = "ℹ️"
                        budget_info = ""
                    
                    print(f"   {status_icon} {stage}: {avg_ms:.1f}ms{budget_info}")
        else:
            # Print summary
            print(f"   📱 Backend: {frame.backend_name}")
            print(f"   🎯 Window: {window.width}x{window.height}")
            print(f"   📸 Frame: {width}x{height}")
            print(f"   ⏱️ Total time: {total_duration:.1f}ms")
        
        print("\n✅ Live smoke test completed successfully!")
        
        # Print enhanced diagnostics if available
        if args.diag:
            print("\n🔍 Enhanced Live Diagnostics:")
            print("=" * 40)
            diagnostics.print_diagnostics(detailed=True)
        
    except Exception as e:
        print(f"\n❌ Live smoke test failed: {e}")
        logger.error(f"Live smoke test error: {e}")
        
        # Record error in diagnostics
        diagnostics.record_error(str(e), "live_smoke_test")
        
        # Still try to save error info if debug active
        if debug_manager:
            error_metadata = {
                'error': str(e),
                'performance_stats': tracker.get_performance_stats(),
                'diagnostic_status': diagnostics.get_current_status(),
                'test_failed': True
            }
            debug_manager.save_json(error_metadata, "live_smoke_error")
    
    finally:
        # End debug dump
        if debug_manager:
            debug_manager.end_dump()
            debug_dir = debug_manager.get_current_dump_dir()
            print(f"\n🐛 Debug artifacts saved to: {debug_dir}")
        
        print("\n🧪 Live smoke test completed.")

def run_ui_doctor_from_cli(args) -> None:
    """
    Run UI Doctor mode from CLI.
    
    Args:
        args: Parsed command line arguments containing ui_doctor and ui_safe_demo flags
    """
    from pathlib import Path
    import sys
    import os
    from datetime import datetime
    from .utils.debug_dump import begin_run, end_run, dump_json
    from .ui.ui_health import UIHealthReporter, take_window_screenshot, detect_uniform_frame
    
    print("🩺 Starting UI Doctor diagnostic mode...")
    
    # Create debug run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_tag = getattr(args, 'debug_tag', None) or "ui_doctor"
    run_id = f"{timestamp}_{debug_tag}"
    
    try:
        # Check if we can run with a display
        if not _can_run_gui_tests():
            print("❌ No display available — UI doctor skipped")
            sys.exit(2)
        
        # Start debug run
        with begin_run(run_id):
            # Initialize and run GUI in Safe Demo mode for diagnostics
            from integrated_arena_bot_gui import IntegratedArenaBotGUI
            
            print("🚀 Launching GUI in diagnostic mode...")
            bot = IntegratedArenaBotGUI(ui_safe_demo=True)
            
            # Give GUI time to initialize and render
            import time
            bot.root.after(2000, lambda: _ui_doctor_analyze_and_exit(bot, run_id))
            bot.root.after(5000, bot.root.quit)  # Force exit after 5 seconds
            
            # Run GUI for a short time
            bot.root.mainloop()
            
    except Exception as e:
        print(f"❌ UI Doctor failed: {e}")
        sys.exit(1)


def _can_run_gui_tests() -> bool:
    """Check if GUI tests can be run (display available)."""
    try:
        import os
        import tkinter as tk
        
        # Check for display environment
        if os.name == 'nt':  # Windows
            return True
        elif 'DISPLAY' in os.environ:  # Linux with X11
            # Try to create a simple tkinter window
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            return True
        else:
            return False
            
    except Exception:
        return False


def _ui_doctor_analyze_and_exit(bot, run_id: str):
    """Analyze UI health and take diagnostic screenshots."""
    from pathlib import Path
    import time
    from .utils.debug_dump import get_debug_run_dir, dump_json
    from .ui.ui_health import take_window_screenshot, detect_uniform_frame
    
    try:
        debug_dir = get_debug_run_dir(run_id)
        
        # Get UI health report
        if bot.ui_health_reporter:
            ui_health = bot.ui_health_reporter.get_ui_health_report()
            health_summary = bot.ui_health_reporter.get_one_line_summary()
            print(health_summary)
            
            # Dump health report
            health_path = debug_dir / "ui_health.json"
            dump_json(ui_health, str(health_path))
            
            # Take screenshot
            screenshot_path = debug_dir / "ui_snapshot.png"
            if take_window_screenshot(bot.root, screenshot_path):
                print(f"📸 Window screenshot saved: {screenshot_path}")
                
                # Analyze for uniform fill
                uniform_stats = detect_uniform_frame(screenshot_path)
                
                # Dump uniform detection stats
                stats_path = debug_dir / "ui_uniform_stats.json"
                dump_json(uniform_stats, str(stats_path))
                
                # Print doctor results
                paint_count = ui_health.get('paint_counter', 0)
                variance = uniform_stats.get('statistics', {}).get('grayscale', {}).get('variance', 0)
                uniform_detected = uniform_stats.get('uniform_detected', False)
                
                central_widget = ui_health.get('central_widget', {})
                has_layout = central_widget.get('has_layout', False)
                
                # Print diagnostic summary
                print(f"🩺 UI Doctor Results:")
                print(f"   Paint count: {paint_count}")
                print(f"   Frame variance: {variance:.2f}")
                print(f"   Layout present: {'yes' if has_layout else 'NO'}")
                print(f"   Uniform fill: {'YES' if uniform_detected else 'no'}")
                print(f"   Debug artifacts: {debug_dir}")
                
                # Check for uniform fill
                if uniform_detected:
                    print("⚠️  UI appears uniform (blue screen) — applying auto-triage fixes...")
                    
                    # Apply auto-triage fixes
                    from ..ui.auto_triage import run_auto_triage
                    triage_result = run_auto_triage(bot.root, bot.ui_health_reporter)
                    
                    # Log triage results
                    triage_path = debug_dir / "ui_auto_triage.json"
                    dump_json(triage_result, str(triage_path))
                    
                    print(f"🔧 Auto-triage applied {len(triage_result['fixes_applied'])} fixes:")
                    for fix in triage_result['fixes_applied']:
                        print(f"   - {fix}")
                    
                    # Re-test after fixes
                    bot.root.update()
                    time.sleep(0.5)
                    
                    retest_screenshot_path = debug_dir / "ui_snapshot_after_fixes.png"
                    if take_window_screenshot(bot.root, retest_screenshot_path):
                        retest_stats = detect_uniform_frame(retest_screenshot_path)
                        retest_uniform = retest_stats.get('uniform_detected', True)
                        retest_variance = retest_stats.get('statistics', {}).get('grayscale', {}).get('variance', 0)
                        
                        # Update diagnosis
                        triage_result['retest_uniform_detected'] = retest_uniform
                        triage_result['retest_variance'] = retest_variance
                        triage_result['uniform_fill_resolved'] = not retest_uniform
                        
                        # Re-dump with retest results
                        dump_json(triage_result, str(triage_path))
                        
                        if not retest_uniform:
                            print(f"✅ Auto-triage successful! Variance improved to {retest_variance:.2f}")
                            # Schedule exit with success code
                            bot.root.after(100, lambda: _exit_with_code(0))
                        else:
                            print(f"❌ Auto-triage failed. Variance still {retest_variance:.2f}")
                            print("   Manual investigation required:")
                            print("   - Check for threading issues in paint events")
                            print("   - Verify widget z-order and visibility")
                            print("   - Review custom painting code")
                            # Schedule exit with error code
                            bot.root.after(100, lambda: _exit_with_code(1))
                    else:
                        print("❌ Could not retest after auto-triage")
                        bot.root.after(100, lambda: _exit_with_code(1))
                else:
                    print("✅ UI appears to be rendering correctly")
                    # Schedule exit with success code
                    bot.root.after(100, lambda: _exit_with_code(0))
            else:
                print("❌ Failed to take screenshot")
                bot.root.after(100, lambda: _exit_with_code(1))
        else:
            print("❌ UI health reporter not available")
            bot.root.after(100, lambda: _exit_with_code(1))
            
    except Exception as e:
        print(f"❌ UI Doctor analysis failed: {e}")
        bot.root.after(100, lambda: _exit_with_code(1))


def _exit_with_code(code: int):
    """Exit with the specified code."""
    import sys
    sys.exit(code)
