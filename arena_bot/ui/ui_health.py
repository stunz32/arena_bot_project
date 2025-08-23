"""
UI Health Reporter

Collects and reports UI health metrics for diagnostic purposes.
"""

import json
import tkinter as tk
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import time
from pathlib import Path


class UIHealthReporter:
    """Collects UI health metrics and diagnostic information."""
    
    def __init__(self, root_window: Optional[tk.Tk] = None):
        """
        Initialize UI health reporter.
        
        Args:
            root_window: Main tkinter window to monitor
        """
        self.root_window = root_window
        self.paint_counter = 0
        self._lock = threading.Lock()
        self.start_time = time.time()
        
    def increment_paint_counter(self):
        """Increment paint counter safely."""
        with self._lock:
            self.paint_counter += 1
    
    def get_ui_health_report(self) -> Dict[str, Any]:
        """
        Collect comprehensive UI health metrics.
        
        Returns:
            Dictionary containing UI health metrics
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'paint_counter': self.paint_counter,
            'window_available': self.root_window is not None
        }
        
        if self.root_window:
            try:
                # Main window information
                report['main_window'] = {
                    'class': type(self.root_window).__name__,
                    'wm_state': self.root_window.wm_state(),
                    'geometry': self.root_window.winfo_geometry(),
                    'width': self.root_window.winfo_width(),
                    'height': self.root_window.winfo_height(),
                    'x': self.root_window.winfo_x(),
                    'y': self.root_window.winfo_y(),
                    'toplevel': self.root_window.winfo_toplevel() is not None
                }
                
                # Central widget information
                central_widget = None
                children = self.root_window.winfo_children()
                report['main_window']['child_count'] = len(children)
                
                # Look for main content frame or similar
                main_content = None
                for child in children:
                    if hasattr(child, '_name') and 'content' in getattr(child, '_name', '').lower():
                        main_content = child
                        break
                    elif hasattr(child, 'winfo_name') and 'content' in child.winfo_name().lower():
                        main_content = child
                        break
                
                if main_content:
                    report['central_widget'] = {
                        'exists': True,
                        'class': type(main_content).__name__,
                        'has_layout': hasattr(main_content, 'pack_slaves') or hasattr(main_content, 'grid_slaves'),
                        'layout_item_count': len(getattr(main_content, 'pack_slaves', lambda: [])()) + len(getattr(main_content, 'grid_slaves', lambda: [])()),
                        'visible': main_content.winfo_viewable()
                    }
                else:
                    report['central_widget'] = {
                        'exists': False,
                        'has_layout': False,
                        'layout_item_count': 0
                    }
                
                # Device pixel ratio (approximation for tkinter)
                report['display_info'] = {
                    'device_pixel_ratio': self.root_window.tk.call('tk', 'scaling'),
                    'screen_width': self.root_window.winfo_screenwidth(),
                    'screen_height': self.root_window.winfo_screenheight(),
                    'depth': self.root_window.winfo_depth()
                }
                
                # Window flags and properties
                try:
                    attributes = {}
                    for attr in ['-alpha', '-disabled', '-fullscreen', '-modified', '-notify', '-titlepath', '-topmost', '-transparent', '-type', '-zoomed']:
                        try:
                            attributes[attr.lstrip('-')] = self.root_window.wm_attributes(attr)
                        except tk.TclError:
                            pass  # Attribute not supported on this platform
                    report['window_attributes'] = attributes
                except Exception as e:
                    report['window_attributes'] = {'error': str(e)}
                
                # Stylesheet summary (tkinter doesn't have stylesheets like Qt, but we can check configure options)
                try:
                    config = self.root_window.config()
                    stylesheet_summary = {
                        'background': config.get('background', ['', '', '', '', 'SystemButtonFace'])[4],
                        'relief': config.get('relief', ['', '', '', '', 'flat'])[4],
                        'borderwidth': config.get('borderwidth', ['', '', '', '', '0'])[4]
                    }
                    report['stylesheet_summary'] = stylesheet_summary
                except Exception as e:
                    report['stylesheet_summary'] = {'error': str(e)}
                
            except Exception as e:
                report['window_error'] = str(e)
                report['main_window'] = {'error': str(e)}
                report['central_widget'] = {'error': str(e)}
        
        return report
    
    def dump_ui_health_report(self, output_dir: Path, run_id: str = None) -> Path:
        """
        Dump UI health report to JSON file.
        
        Args:
            output_dir: Directory to write the report
            run_id: Optional run identifier for filename
            
        Returns:
            Path to the written file
        """
        report = self.get_ui_health_report()
        
        if run_id:
            filename = f"{run_id}_ui_health.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ui_health_{timestamp}.json"
        
        output_path = output_dir / filename
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def get_one_line_summary(self) -> str:
        """
        Get one-line summary of UI health status.
        
        Returns:
            Human-readable status line
        """
        report = self.get_ui_health_report()
        
        if not report['window_available']:
            return "❌ UI: No window available"
        
        paint_count = report['paint_counter']
        uptime = int(report['uptime_seconds'])
        
        status_parts = [f"paints:{paint_count}", f"uptime:{uptime}s"]
        
        # Central widget status
        if 'central_widget' in report:
            if report['central_widget']['exists']:
                layout_count = report['central_widget']['layout_item_count']
                status_parts.append(f"layout:{layout_count}items")
                
                if report['central_widget']['has_layout']:
                    status_parts.append("layout:yes")
                else:
                    status_parts.append("layout:NO")
            else:
                status_parts.append("central:MISSING")
        
        # Window geometry
        if 'main_window' in report and 'width' in report['main_window']:
            width = report['main_window']['width']
            height = report['main_window']['height']
            status_parts.append(f"size:{width}x{height}")
        
        return f"✅ UI: {', '.join(status_parts)}"


def take_window_screenshot(root_window: tk.Tk, output_path: Path) -> bool:
    """
    Take a screenshot of the tkinter window.
    
    Args:
        root_window: Tkinter window to screenshot
        output_path: Path to save screenshot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import PIL.ImageGrab
        
        # Get window position and size
        root_window.update_idletasks()
        x = root_window.winfo_rootx()
        y = root_window.winfo_rooty()
        width = root_window.winfo_width()
        height = root_window.winfo_height()
        
        # Capture window area
        screenshot = PIL.ImageGrab.grab(bbox=(x, y, x + width, y + height))
        screenshot.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"Failed to take window screenshot: {e}")
        return False


def detect_uniform_frame(screenshot_path: Path) -> Dict[str, Any]:
    """
    Analyze screenshot for uniform fill (blue screen detection).
    
    Args:
        screenshot_path: Path to screenshot image
        
    Returns:
        Dictionary with uniform detection results
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Load image
        with Image.open(screenshot_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            width, height = img.size
            
            # Calculate statistics per channel
            stats = {}
            uniform_detected = False
            
            # Check each color channel
            for i, channel in enumerate(['red', 'green', 'blue']):
                channel_data = img_array[:, :, i].flatten()
                
                channel_stats = {
                    'min': int(np.min(channel_data)),
                    'max': int(np.max(channel_data)),
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'variance': float(np.var(channel_data))
                }
                stats[channel] = channel_stats
            
            # Overall grayscale statistics
            gray = np.mean(img_array, axis=2)
            overall_stats = {
                'min': float(np.min(gray)),
                'max': float(np.max(gray)),
                'mean': float(np.mean(gray)),
                'std': float(np.std(gray)),
                'variance': float(np.var(gray))
            }
            stats['grayscale'] = overall_stats
            
            # Detect uniform fill (low variance indicates uniform color)
            variance_threshold = 100.0  # Adjust as needed
            uniform_detected = overall_stats['variance'] < variance_threshold
            
            result = {
                'screenshot_path': str(screenshot_path),
                'image_size': {'width': width, 'height': height},
                'uniform_detected': uniform_detected,
                'variance_threshold': variance_threshold,
                'statistics': stats,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return result
            
    except Exception as e:
        return {
            'screenshot_path': str(screenshot_path),
            'error': str(e),
            'uniform_detected': False,
            'analysis_timestamp': datetime.now().isoformat()
        }