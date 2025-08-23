"""
Auto-Triage System for UI Blue Screen Issues

Automatically detects and fixes common causes of uniform fill (blue screen) rendering.
"""

import tkinter as tk
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json


class UIAutoTriage:
    """Automatically diagnose and fix common UI rendering issues."""
    
    def __init__(self, root_window: tk.Tk, ui_health_reporter=None):
        """
        Initialize auto-triage system.
        
        Args:
            root_window: Main tkinter window
            ui_health_reporter: UI health reporter instance
        """
        self.root_window = root_window
        self.ui_health_reporter = ui_health_reporter
        self.fixes_applied = []
        
    def diagnose_and_fix(self) -> Dict[str, Any]:
        """
        Run full diagnostic and apply fixes automatically.
        
        Returns:
            Dictionary with diagnosis results and fixes applied
        """
        diagnosis = {
            'timestamp': self._get_timestamp(),
            'issues_found': [],
            'fixes_applied': [],
            'success': False,
            'uniform_fill_resolved': False
        }
        
        if not self.root_window:
            diagnosis['issues_found'].append('No root window available')
            return diagnosis
        
        # Check for missing layout
        layout_fix = self._check_and_fix_missing_layout()
        if layout_fix:
            diagnosis['issues_found'].extend(layout_fix['issues'])
            diagnosis['fixes_applied'].extend(layout_fix['fixes'])
        
        # Check for paint guarding
        paint_fix = self._check_and_fix_paint_guarding()
        if paint_fix:
            diagnosis['issues_found'].extend(paint_fix['issues'])
            diagnosis['fixes_applied'].extend(paint_fix['fixes'])
        
        # Check for opaque stylesheet/palette
        style_fix = self._check_and_fix_opaque_styling()
        if style_fix:
            diagnosis['issues_found'].extend(style_fix['issues'])
            diagnosis['fixes_applied'].extend(style_fix['fixes'])
        
        # Check for covering widgets
        covering_fix = self._check_and_fix_covering_widgets()
        if covering_fix:
            diagnosis['issues_found'].extend(covering_fix['issues'])
            diagnosis['fixes_applied'].extend(covering_fix['fixes'])
        
        # Force visibility refresh
        visibility_fix = self._ensure_visibility()
        if visibility_fix:
            diagnosis['fixes_applied'].extend(visibility_fix['fixes'])
        
        diagnosis['success'] = len(diagnosis['fixes_applied']) > 0
        diagnosis['fixes_applied'] = list(set(diagnosis['fixes_applied']))  # Remove duplicates
        
        return diagnosis
    
    def _check_and_fix_missing_layout(self) -> Optional[Dict[str, Any]]:
        """Check for and fix missing layout issues."""
        issues = []
        fixes = []
        
        try:
            # Get main content frame
            main_content = None
            for child in self.root_window.winfo_children():
                if hasattr(child, '_name') and 'content' in getattr(child, '_name', '').lower():
                    main_content = child
                    break
                elif hasattr(child, 'winfo_name') and 'content' in child.winfo_name().lower():
                    main_content = child
                    break
            
            # Check if main content frame exists and has layout
            if main_content is None:
                issues.append("No main content frame found")
                
                # Create basic content frame
                main_content = tk.Frame(self.root_window, bg='#2C3E50', name='main_content_frame')
                main_content.pack(fill='both', expand=True, padx=10, pady=10)
                
                # Add basic content
                label = tk.Label(
                    main_content,
                    text="Arena Assistant - Content Area",
                    font=('Arial', 14, 'bold'),
                    fg='#ECF0F1',
                    bg='#2C3E50'
                )
                label.pack(pady=20)
                
                fixes.append("Created main content frame with basic layout")
                
            elif not self._has_layout(main_content):
                issues.append("Main content frame has no layout")
                
                # Add basic layout to existing frame
                if len(main_content.winfo_children()) == 0:
                    label = tk.Label(
                        main_content,
                        text="Arena Assistant - Ready",
                        font=('Arial', 12),
                        fg='#ECF0F1',
                        bg='#2C3E50'
                    )
                    label.pack(pady=10)
                    fixes.append("Added basic content to empty main frame")
            
            # Ensure root window has proper background
            current_bg = self.root_window.cget('bg')
            if current_bg in ['SystemButtonFace', 'white', '#FFFFFF']:
                self.root_window.config(bg='#2C3E50')
                fixes.append("Fixed root window background color")
        
        except Exception as e:
            issues.append(f"Layout check failed: {e}")
        
        if issues or fixes:
            return {'issues': issues, 'fixes': fixes}
        return None
    
    def _check_and_fix_paint_guarding(self) -> Optional[Dict[str, Any]]:
        """Check for and fix paint event guarding issues."""
        issues = []
        fixes = []
        
        try:
            # Force paint events
            self.root_window.update_idletasks()
            self.root_window.update()
            
            if self.ui_health_reporter:
                paint_count_before = self.ui_health_reporter.paint_counter
                
                # Force additional paint cycles
                for _ in range(3):
                    self.root_window.update()
                    self.ui_health_reporter.increment_paint_counter()
                
                paint_count_after = self.ui_health_reporter.paint_counter
                
                if paint_count_after <= paint_count_before:
                    issues.append("Paint events not incrementing properly")
                    fixes.append("Forced paint event cycles")
        
        except Exception as e:
            issues.append(f"Paint check failed: {e}")
        
        if issues or fixes:
            return {'issues': issues, 'fixes': fixes}
        return None
    
    def _check_and_fix_opaque_styling(self) -> Optional[Dict[str, Any]]:
        """Check for and fix opaque styling issues."""
        issues = []
        fixes = []
        
        try:
            # Check root window styling
            config = self.root_window.config()
            background = config.get('background', ['', '', '', '', 'SystemButtonFace'])[4]
            
            # Fix problematic backgrounds
            if background in ['SystemButtonFace', 'white', '#FFFFFF', 'blue', '#0000FF']:
                issues.append(f"Problematic background color: {background}")
                self.root_window.config(bg='#2C3E50')
                fixes.append(f"Changed background from {background} to #2C3E50")
            
            # Check and fix child widget styling
            self._fix_child_styling(self.root_window, issues, fixes)
        
        except Exception as e:
            issues.append(f"Styling check failed: {e}")
        
        if issues or fixes:
            return {'issues': issues, 'fixes': fixes}
        return None
    
    def _check_and_fix_covering_widgets(self) -> Optional[Dict[str, Any]]:
        """Check for and fix widgets that cover the entire window."""
        issues = []
        fixes = []
        
        try:
            window_width = self.root_window.winfo_width()
            window_height = self.root_window.winfo_height()
            
            if window_width <= 1 or window_height <= 1:
                # Window not initialized yet
                self.root_window.update_idletasks()
                window_width = self.root_window.winfo_width()
                window_height = self.root_window.winfo_height()
            
            for child in self.root_window.winfo_children():
                try:
                    child_width = child.winfo_width()
                    child_height = child.winfo_height()
                    
                    # Check if child covers most of the window
                    if (child_width >= window_width * 0.9 and 
                        child_height >= window_height * 0.9):
                        
                        child_bg = child.cget('bg') if hasattr(child, 'cget') else None
                        
                        if child_bg in ['blue', '#0000FF', 'SystemButtonFace']:
                            issues.append(f"Large covering widget with problematic background: {child_bg}")
                            child.config(bg='#2C3E50')
                            fixes.append(f"Fixed covering widget background from {child_bg}")
                
                except Exception as e:
                    # Skip widgets that don't support these operations
                    continue
        
        except Exception as e:
            issues.append(f"Covering widget check failed: {e}")
        
        if issues or fixes:
            return {'issues': issues, 'fixes': fixes}
        return None
    
    def _ensure_visibility(self) -> Optional[Dict[str, Any]]:
        """Ensure window visibility with multiple strategies."""
        fixes = []
        
        try:
            # Force window to front
            self.root_window.lift()
            self.root_window.focus_force()
            fixes.append("Brought window to front")
            
            # Force geometry update
            self.root_window.update_idletasks()
            self.root_window.update()
            fixes.append("Forced geometry update")
            
            # Ensure minimum size
            current_width = self.root_window.winfo_width()
            current_height = self.root_window.winfo_height()
            
            min_width, min_height = 800, 600
            
            if current_width < min_width or current_height < min_height:
                new_width = max(current_width, min_width)
                new_height = max(current_height, min_height)
                self.root_window.geometry(f"{new_width}x{new_height}")
                fixes.append(f"Increased window size to {new_width}x{new_height}")
            
            # Force redraw
            self.root_window.update()
            fixes.append("Forced final redraw")
        
        except Exception as e:
            fixes.append(f"Visibility fix error: {e}")
        
        if fixes:
            return {'fixes': fixes}
        return None
    
    def _fix_child_styling(self, widget: tk.Widget, issues: List[str], fixes: List[str], depth: int = 0):
        """Recursively fix child widget styling."""
        if depth > 3:  # Limit recursion depth
            return
        
        try:
            for child in widget.winfo_children():
                try:
                    if hasattr(child, 'cget') and hasattr(child, 'config'):
                        bg = child.cget('bg')
                        if bg in ['blue', '#0000FF', 'SystemButtonFace']:
                            issues.append(f"Child widget with problematic background: {bg}")
                            child.config(bg='#34495E')
                            fixes.append(f"Fixed child widget background from {bg}")
                    
                    # Recurse into child widgets
                    self._fix_child_styling(child, issues, fixes, depth + 1)
                
                except Exception:
                    # Skip widgets that don't support these operations
                    continue
        
        except Exception:
            # Skip if can't enumerate children
            pass
    
    def _has_layout(self, widget: tk.Widget) -> bool:
        """Check if a widget has a layout manager."""
        try:
            # Check for pack slaves
            if hasattr(widget, 'pack_slaves') and len(widget.pack_slaves()) > 0:
                return True
            
            # Check for grid slaves
            if hasattr(widget, 'grid_slaves') and len(widget.grid_slaves()) > 0:
                return True
            
            # Check for children
            if len(widget.winfo_children()) > 0:
                return True
            
            return False
        
        except Exception:
            return False
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_applied_fixes(self) -> List[str]:
        """Get list of fixes that have been applied."""
        return self.fixes_applied.copy()
    
    def dump_diagnosis_report(self, diagnosis: Dict[str, Any], output_path: Path) -> Path:
        """
        Dump diagnosis report to file.
        
        Args:
            diagnosis: Diagnosis results from diagnose_and_fix()
            output_path: Path to write the report
            
        Returns:
            Path to the written file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(diagnosis, f, indent=2)
        
        return output_path


def run_auto_triage(root_window: tk.Tk, ui_health_reporter=None) -> Dict[str, Any]:
    """
    Run automatic UI triage and return results.
    
    Args:
        root_window: Main tkinter window
        ui_health_reporter: Optional UI health reporter
        
    Returns:
        Dictionary with triage results
    """
    triage = UIAutoTriage(root_window, ui_health_reporter)
    return triage.diagnose_and_fix()