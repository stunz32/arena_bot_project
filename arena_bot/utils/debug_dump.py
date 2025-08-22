"""
Black-box debug dump system for Arena Bot pipeline observability.

This module provides a lightweight debug dump system that captures pipeline artifacts,
timing data, and failure context to make bugs impossible to hide. Every pipeline
decision becomes observable through automatic dumps.

Features:
- Timestamped debug run directories
- Image and JSON artifact dumping
- Stage-wise failure tracking
- Performance timing capture
- Thread-safe operations

Usage:
    from arena_bot.utils.debug_dump import begin_run, dump_image, dump_json, end_run
    
    # Start a debug run
    run_dir = begin_run("detection_pipeline")
    
    # Dump artifacts throughout pipeline
    dump_image(screenshot, "input_frame")
    dump_json({"candidates": matches, "confidence": 0.85}, "detection_results")
    
    # End the run
    end_run()
"""

import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import threading
import logging

try:
    import numpy as np
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)

# Global state for current debug run (thread-local)
_thread_local = threading.local()

# Configuration
DEBUG_BASE_DIR = Path(".debug_runs")
MAX_DEBUG_RUNS = 50  # Keep last N runs
DEFAULT_IMAGE_QUALITY = 85  # JPEG quality for debug images


class DebugDumpError(Exception):
    """Raised when debug dump operations fail"""
    pass


def _get_current_run() -> Optional[Path]:
    """Get current debug run directory (thread-local)"""
    return getattr(_thread_local, 'current_run_dir', None)


def _set_current_run(run_dir: Optional[Path]) -> None:
    """Set current debug run directory (thread-local)"""
    _thread_local.current_run_dir = run_dir


def _ensure_debug_dir() -> Path:
    """Ensure debug base directory exists and clean up old runs"""
    DEBUG_BASE_DIR.mkdir(exist_ok=True)
    
    # Clean up old runs if we have too many
    existing_runs = sorted(DEBUG_BASE_DIR.glob("*_*"), key=lambda p: p.stat().st_mtime)
    if len(existing_runs) >= MAX_DEBUG_RUNS:
        # Remove oldest runs
        for old_run in existing_runs[:-MAX_DEBUG_RUNS + 1]:
            try:
                import shutil
                shutil.rmtree(old_run)
                logger.debug(f"Cleaned up old debug run: {old_run.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up old debug run {old_run}: {e}")
    
    return DEBUG_BASE_DIR


def begin_run(tag: str) -> Path:
    """
    Begin a new debug run with the specified tag.
    
    Args:
        tag: Human-readable tag for this debug run (e.g., "detection_pipeline")
        
    Returns:
        Path to the debug run directory
        
    Raises:
        DebugDumpError: If unable to create debug directory
    """
    if not tag or not isinstance(tag, str):
        raise DebugDumpError("Debug run tag must be a non-empty string")
    
    # Clean tag for filesystem
    clean_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    run_name = f"{timestamp}_{clean_tag}_{run_id}"
    
    try:
        debug_base = _ensure_debug_dir()
        run_dir = debug_base / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set as current run
        _set_current_run(run_dir)
        
        # Create run metadata
        metadata = {
            "tag": tag,
            "start_time": datetime.now().isoformat(),
            "run_id": run_id,
            "thread_id": threading.get_ident(),
            "artifacts": []
        }
        
        metadata_file = run_dir / "run_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Started debug run: {run_name}")
        return run_dir
        
    except Exception as e:
        raise DebugDumpError(f"Failed to create debug run directory: {e}") from e


def dump_image(
    image_data: Union[str, Path, np.ndarray], 
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Path]:
    """
    Dump an image to the current debug run.
    
    Args:
        image_data: Image file path, numpy array, or PIL image
        name: Name for the dumped image (without extension)
        metadata: Optional metadata dict to save alongside image
        
    Returns:
        Path to saved image file, or None if no active debug run
        
    Raises:
        DebugDumpError: If image dump fails
    """
    run_dir = _get_current_run()
    if not run_dir:
        logger.debug("No active debug run - skipping image dump")
        return None
    
    if not name or not isinstance(name, str):
        raise DebugDumpError("Image name must be a non-empty string")
    
    # Clean name for filesystem
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    
    try:
        # Handle different image data types
        if isinstance(image_data, (str, Path)):
            # Copy existing image file
            src_path = Path(image_data)
            if not src_path.exists():
                raise DebugDumpError(f"Source image file not found: {src_path}")
            
            # Determine extension from source
            ext = src_path.suffix or ".png"
            dst_path = run_dir / f"{clean_name}{ext}"
            
            import shutil
            shutil.copy2(src_path, dst_path)
            
        elif HAS_CV2 and isinstance(image_data, np.ndarray):
            # Save numpy array as image
            dst_path = run_dir / f"{clean_name}.jpg"
            
            # Convert to BGR if needed (OpenCV expects BGR)
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # Assume RGB, convert to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_data
            
            success = cv2.imwrite(str(dst_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_IMAGE_QUALITY])
            if not success:
                raise DebugDumpError(f"OpenCV failed to write image: {dst_path}")
                
        else:
            raise DebugDumpError(f"Unsupported image data type: {type(image_data)}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = run_dir / f"{clean_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Update run metadata
        _update_run_artifacts("image", clean_name, {
            "file": dst_path.name,
            "metadata_file": f"{clean_name}_metadata.json" if metadata else None,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.debug(f"Dumped image: {dst_path.name}")
        return dst_path
        
    except Exception as e:
        raise DebugDumpError(f"Failed to dump image '{name}': {e}") from e


def dump_json(obj: Dict[str, Any], name: str) -> Optional[Path]:
    """
    Dump a JSON object to the current debug run.
    
    Args:
        obj: Dictionary object to serialize as JSON
        name: Name for the JSON file (without extension)
        
    Returns:
        Path to saved JSON file, or None if no active debug run
        
    Raises:
        DebugDumpError: If JSON dump fails
    """
    run_dir = _get_current_run()
    if not run_dir:
        logger.debug("No active debug run - skipping JSON dump")
        return None
    
    if not name or not isinstance(name, str):
        raise DebugDumpError("JSON name must be a non-empty string")
    
    if not isinstance(obj, dict):
        raise DebugDumpError("JSON object must be a dictionary")
    
    # Clean name for filesystem
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    
    try:
        json_path = run_dir / f"{clean_name}.json"
        
        # Add timestamp to object
        obj_with_meta = {
            **obj,
            "_debug_timestamp": datetime.now().isoformat(),
            "_debug_thread_id": threading.get_ident()
        }
        
        with open(json_path, 'w') as f:
            json.dump(obj_with_meta, f, indent=2, default=str)
        
        # Update run metadata
        _update_run_artifacts("json", clean_name, {
            "file": json_path.name,
            "timestamp": datetime.now().isoformat(),
            "keys": list(obj.keys())
        })
        
        logger.debug(f"Dumped JSON: {json_path.name}")
        return json_path
        
    except Exception as e:
        raise DebugDumpError(f"Failed to dump JSON '{name}': {e}") from e


def _update_run_artifacts(artifact_type: str, name: str, details: Dict[str, Any]) -> None:
    """Update run metadata with new artifact"""
    run_dir = _get_current_run()
    if not run_dir:
        return
    
    metadata_file = run_dir / "run_metadata.json"
    try:
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"artifacts": []}
        
        artifact_info = {
            "type": artifact_type,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        
        metadata.setdefault("artifacts", []).append(artifact_info)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        logger.warning(f"Failed to update run metadata: {e}")


def end_run() -> None:
    """
    End the current debug run and finalize metadata.
    """
    run_dir = _get_current_run()
    if not run_dir:
        logger.debug("No active debug run to end")
        return
    
    try:
        # Finalize metadata
        metadata_file = run_dir / "run_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata["end_time"] = datetime.now().isoformat()
            metadata["duration_ms"] = (
                datetime.fromisoformat(metadata["end_time"]) - 
                datetime.fromisoformat(metadata["start_time"])
            ).total_seconds() * 1000
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Ended debug run: {run_dir.name}")
        
    except Exception as e:
        logger.warning(f"Failed to finalize debug run: {e}")
    
    finally:
        # Clear current run
        _set_current_run(None)


@contextmanager
def debug_run(tag: str):
    """
    Context manager for debug runs.
    
    Usage:
        with debug_run("detection_pipeline"):
            dump_image(frame, "input")
            # ... pipeline operations ...
            dump_json(results, "output")
    """
    run_dir = begin_run(tag)
    try:
        yield run_dir
    finally:
        end_run()


def get_current_run_dir() -> Optional[Path]:
    """Get the current debug run directory, if any"""
    return _get_current_run()


def is_debug_active() -> bool:
    """Check if a debug run is currently active"""
    return _get_current_run() is not None


# Convenience functions for pipeline integration
def dump_detection_failure(
    input_frame: np.ndarray,
    candidates: List[Dict[str, Any]],
    confidence_threshold: float,
    stage: str,
    error_msg: str
) -> None:
    """
    Convenience function to dump detection pipeline failures.
    
    Args:
        input_frame: Input image that failed detection
        candidates: List of candidate matches with metadata
        confidence_threshold: Minimum confidence threshold
        stage: Pipeline stage that failed (e.g., "histogram_match")
        error_msg: Error description
    """
    if not is_debug_active():
        return
    
    # Dump input frame
    dump_image(input_frame, f"failure_{stage}_input")
    
    # Dump failure context
    failure_context = {
        "stage": stage,
        "error": error_msg,
        "confidence_threshold": confidence_threshold,
        "num_candidates": len(candidates),
        "candidates": candidates[:5],  # Top 5 to avoid huge dumps
        "timestamp": datetime.now().isoformat()
    }
    
    dump_json(failure_context, f"failure_{stage}_context")


def dump_stage_timing(stage: str, duration_ms: float, metadata: Optional[Dict] = None) -> None:
    """
    Convenience function to dump stage timing information.
    
    Args:
        stage: Pipeline stage name
        duration_ms: Stage execution time in milliseconds  
        metadata: Optional additional metadata
    """
    if not is_debug_active():
        return
    
    timing_data = {
        "stage": stage,
        "duration_ms": duration_ms,
        "timestamp": datetime.now().isoformat(),
        **(metadata or {})
    }
    
    dump_json(timing_data, f"timing_{stage}")