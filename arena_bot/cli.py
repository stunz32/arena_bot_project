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