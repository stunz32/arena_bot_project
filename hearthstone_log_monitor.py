#!/usr/bin/env python3
"""
Hearthstone Log Monitor - Arena Tracker Style
Monitors Hearthstone log files for real-time game state detection.
Based on Arena Tracker's proven log monitoring methodology.
"""

import os
import sys
import time
import re
import hashlib  # NEW: For event deduplication signatures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum

try:
    import winreg  # For Windows registry access
except ImportError:
    winreg = None  # Not available on non-Windows platforms
    
from logging_config import setup_logging, get_platform_info

class GameState(Enum):
    """Game state detection from logs."""
    UNKNOWN = "Unknown"
    LOGIN = "Login Screen"
    HUB = "Main Menu"
    ARENA_DRAFT = "Arena Draft"
    DRAFT_COMPLETE = "Draft Complete"
    IN_GAME = "In Game"
    GAMEPLAY = "Playing Match"
    COLLECTION = "Collection"
    TOURNAMENT = "Tournament"
    BATTLEGROUNDS = "Battlegrounds"
    ADVENTURE = "Adventure"
    TAVERN_BRAWL = "Tavern Brawl"
    SHOP = "Shop"

@dataclass
class DraftPick:
    """Arena draft pick information."""
    slot: int
    card_code: str
    is_premium: bool
    timestamp: datetime

@dataclass
class LogEntry:
    """Single log entry with metadata."""
    timestamp: datetime
    component: str
    message: str
    raw_line: str

class HearthstoneLogMonitor:
    """
    Monitors Hearthstone log files for real-time game state detection.
    Uses Arena Tracker's proven methodology for maximum accuracy.
    """
    
    def __init__(self, logs_base_path: Optional[str] = None):
        """Initialize the log monitor with intelligent path detection."""
        # Set up logging first
        self.logger = setup_logging(console_output=True)
        
        # Intelligent path resolution
        self.logs_base_path = self._resolve_hearthstone_logs_path(logs_base_path)
        self.platform_info = get_platform_info()
        self.current_log_dir: Optional[Path] = None
        self.log_files: Dict[str, Path] = {}
        self.log_positions: Dict[str, int] = {}
        self.monitoring = False
        
        # Game state
        self.current_game_state = GameState.UNKNOWN
        self.current_draft_picks: List[DraftPick] = []
        self.current_hero: Optional[str] = None
        
        # Callbacks
        self.on_game_state_change = None
        self.on_draft_pick = None
        self.on_draft_start = None
        self.on_draft_complete = None
        
        # NEW: Event deduplication and heartbeat monitoring (AI Helper integration)
        self.event_deduplication_cache = set()  # Cache for preventing duplicate events
        self.max_cache_size = 1000  # Prevent memory growth
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 30  # seconds
        self.log_file_accessible = True
        self.error_recovery_attempts = 0
        self.max_error_recovery_attempts = 3
        
        # Arena Tracker-style regex patterns (Enhanced for AI Helper integration)
        self.patterns = {
            'draft_pick': re.compile(r'DraftManager\.OnChosen.*Slot=(\d+).*cardId=([A-Z0-9_]+).*Premium=(\w+)'),
            'draft_hero': re.compile(r'DraftManager\.OnHeroChosen.*HeroCardID=([A-Z0-9_]+)'),
            'draft_choices': re.compile(r'DraftManager\.OnChoicesAndContents'),
            # NEW: Enhanced draft choices pattern for AI Helper integration
            'draft_choices_detailed': re.compile(r'DraftManager\.OnChoicesAndContents.*(?:cardId=([A-Z0-9_]+).*){3}'),
            'draft_deck_card': re.compile(r'Draft deck contains card ([A-Z0-9_]+)'),
            'scene_loaded': re.compile(r'LoadingScreen\.OnSceneLoaded.*currMode=(\w+)'),
            'scene_unload': re.compile(r'LoadingScreen\.OnScenePreUnload.*nextMode=(\w+)'),
            'asset_load': re.compile(r'AssetLoader.*Loading.*([A-Z0-9_]+)'),
        }
        
        self.logger.info("🎯 Hearthstone Log Monitor Initialized")
        self.logger.info(f"📁 Monitoring: {self.logs_base_path}")
        self.logger.info(f"🖥️ Platform: {self.platform_info['platform']}")
    
    def _generate_event_signature(self, message: str, component: str) -> str:
        """
        Generate a unique signature for event deduplication.
        Uses a combination of message content and timestamp.
        
        Args:
            message: The log message content
            component: The log component (e.g., 'arena', 'power')
            
        Returns:
            str: Unique event signature for deduplication
        """
        # Create a hash of the message content (first 100 chars to avoid long duplicates)
        message_hash = hashlib.md5(message[:100].encode()).hexdigest()[:8]
        return f"{component}:{message_hash}"
    
    def _is_duplicate_event(self, event_signature: str) -> bool:
        """
        Check if an event is a duplicate based on its signature.
        
        Args:
            event_signature: The event signature to check
            
        Returns:
            bool: True if the event is a duplicate, False otherwise
        """
        if event_signature in self.event_deduplication_cache:
            return True
        
        # Add to cache and manage cache size
        self.event_deduplication_cache.add(event_signature)
        
        # Prevent memory growth by clearing old entries
        if len(self.event_deduplication_cache) > self.max_cache_size:
            # Remove the oldest half of entries (approximate)
            old_cache = list(self.event_deduplication_cache)
            self.event_deduplication_cache = set(old_cache[self.max_cache_size // 2:])
        
        return False
    
    def _check_heartbeat_and_log_accessibility(self) -> bool:
        """
        Check if log files are accessible and update heartbeat.
        Implements heartbeat monitoring for log file accessibility.
        
        Returns:
            bool: True if log files are accessible, False otherwise
        """
        now = datetime.now()
        
        # Check if it's time for a heartbeat
        if (now - self.last_heartbeat).seconds >= self.heartbeat_interval:
            self.last_heartbeat = now
            
            # Check log file accessibility with multi-file resilience
            try:
                if self.current_log_dir and self.current_log_dir.exists():
                    # Test multiple files for resilience against temporary locks
                    accessible_files = 0
                    total_files = len(self.log_files)
                    
                    if total_files > 0:
                        for log_type, log_path in self.log_files.items():
                            if log_path.exists() and os.access(log_path, os.R_OK):
                                # Additional test: try to actually read from file to detect locks
                                try:
                                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        f.read(100)  # Read first 100 chars to test actual accessibility
                                    accessible_files += 1
                                    self.logger.debug(f"✅ {log_type} accessible")
                                except (OSError, PermissionError, IOError):
                                    self.logger.debug(f"⚠️ {log_type} read-locked or inaccessible")
                                    continue
                        
                        # Require at least 30% of files accessible (resilient threshold)  
                        accessibility_ratio = accessible_files / total_files
                        if accessibility_ratio >= 0.3:
                            self.log_file_accessible = True
                            self.error_recovery_attempts = 0  # Reset error counter on success
                            self.logger.debug(f"💓 Multi-file check passed - {accessible_files}/{total_files} files accessible ({accessibility_ratio:.1%})")
                        else:
                            self.log_file_accessible = False
                            self.logger.warning(f"💔 Multi-file check failed - Only {accessible_files}/{total_files} files accessible ({accessibility_ratio:.1%})")
                            self._diagnose_heartbeat_failure()
                    else:
                        self.log_file_accessible = False
                        self.logger.error("💔 No log files discovered for accessibility testing")
                else:
                    self.log_file_accessible = False
                    self.logger.error(f"💔 Log directory inaccessible: {self.current_log_dir}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Heartbeat check failed: {e}")
                self.log_file_accessible = False
                self.error_recovery_attempts += 1
                
                # Attempt error recovery if we haven't exceeded max attempts
                if self.error_recovery_attempts <= self.max_error_recovery_attempts:
                    self._attempt_log_error_recovery()
            
            # Log heartbeat status
            if self.log_file_accessible:
                self.logger.info(f"💓 Heartbeat OK - Log files accessible at {now.strftime('%H:%M:%S')}")
            else:
                self.logger.error(f"💔 Heartbeat FAILED - Log files inaccessible at {now.strftime('%H:%M:%S')}")
        
        return self.log_file_accessible
    
    def _diagnose_heartbeat_failure(self):
        """
        Detailed diagnostics for heartbeat failures.
        Provides comprehensive information about what specifically is failing.
        """
        self.logger.error("🔍 HEARTBEAT FAILURE DIAGNOSTICS:")
        self.logger.error(f"  Base path: {self.logs_base_path} (exists: {self.logs_base_path.exists()})")
        
        if self.current_log_dir:
            self.logger.error(f"  Current log dir: {self.current_log_dir} (exists: {self.current_log_dir.exists()})")
            
            if self.current_log_dir.exists():
                try:
                    dir_contents = list(self.current_log_dir.glob("*.log"))
                    self.logger.error(f"  Log files in directory: {len(dir_contents)}")
                    for log_file in dir_contents[:5]:  # Show first 5 files
                        stat = log_file.stat()
                        age_minutes = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 60
                        self.logger.error(f"    {log_file.name}: {stat.st_size} bytes, {age_minutes:.1f}min old")
                except Exception as e:
                    self.logger.error(f"  Error reading log directory: {e}")
        else:
            self.logger.error("  Current log dir: None")
        
        self.logger.error(f"  Discovered log files: {len(self.log_files)}")
        for log_type, log_path in self.log_files.items():
            exists = log_path.exists()
            if exists:
                try:
                    stat = log_path.stat()
                    readable = os.access(log_path, os.R_OK)
                    age_minutes = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 60
                    
                    # Test actual read capability
                    read_test = "UNKNOWN"
                    try:
                        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                            f.read(50)
                        read_test = "SUCCESS"
                    except Exception as e:
                        read_test = f"FAILED ({type(e).__name__})"
                    
                    self.logger.error(f"    {log_type}: exists=✅, size={stat.st_size}B, age={age_minutes:.1f}min, readable={readable}, read_test={read_test}")
                except Exception as e:
                    self.logger.error(f"    {log_type}: exists=✅, stat_error={e}")
            else:
                self.logger.error(f"    {log_type}: exists=❌, path={log_path}")
        
        # Check if this might be a session transition
        if self.logs_base_path.exists():
            try:
                recent_dirs = []
                for item in self.logs_base_path.iterdir():
                    if item.is_dir() and item.name.startswith("Hearthstone_"):
                        age = datetime.now() - datetime.fromtimestamp(item.stat().st_mtime)
                        if age.total_seconds() < 300:  # Within last 5 minutes
                            recent_dirs.append((item, age.total_seconds()))
                
                if recent_dirs:
                    self.logger.error(f"  Recent Hearthstone directories found: {len(recent_dirs)}")
                    for dir_path, age_seconds in recent_dirs:
                        self.logger.error(f"    {dir_path.name}: {age_seconds:.1f}s ago")
                else:
                    self.logger.error("  No recent Hearthstone directories found - possible session transition")
            except Exception as e:
                self.logger.error(f"  Error checking for recent directories: {e}")
    
    def _attempt_log_error_recovery(self):
        """
        Enhanced recovery with detailed diagnostics and multiple recovery strategies.
        Implements intelligent recovery based on failure patterns.
        """
        self.logger.info(f"🔄 Attempting log error recovery (attempt {self.error_recovery_attempts}/{self.max_error_recovery_attempts})")
        
        try:
            # Diagnose the specific issue first
            if not self.logs_base_path.exists():
                self.logger.error(f"❌ Base path doesn't exist: {self.logs_base_path}")
                # Try to re-resolve base path
                old_path = self.logs_base_path
                self.logs_base_path = self._resolve_hearthstone_logs_path()
                self.logger.info(f"🔄 Path resolution: {old_path} → {self.logs_base_path}")
                
                if not self.logs_base_path.exists():
                    self.logger.error("❌ Path re-resolution failed - base path still doesn't exist")
                    return False
            
            # Re-discover log directory with enhanced validation
            old_log_dir = self.current_log_dir
            self.current_log_dir = self.find_latest_log_directory()
            
            if self.current_log_dir:
                self.logger.info(f"🔄 Log directory: {old_log_dir} → {self.current_log_dir}")
                
                # Re-discover log files
                old_file_count = len(self.log_files)
                self.log_files = self.discover_log_files(self.current_log_dir)
                new_file_count = len(self.log_files)
                
                self.logger.info(f"✅ Recovery successful - {old_file_count} → {new_file_count} log files")
                
                # Reset positions to read from current position, not from beginning
                self.log_positions = {}
                
                # Validate that we can actually access the new files
                accessible_files = 0
                for log_type, log_path in self.log_files.items():
                    if log_path.exists() and os.access(log_path, os.R_OK):
                        try:
                            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                f.read(50)  # Test read
                            accessible_files += 1
                        except:
                            continue
                
                if accessible_files > 0:
                    self.logger.info(f"✅ Recovery validation passed - {accessible_files}/{new_file_count} files accessible")
                    return True
                else:
                    self.logger.error(f"❌ Recovery validation failed - no files accessible")
                    return False
            else:
                self.logger.error("❌ Recovery failed - no log directory found")
                
                # Try to diagnose why no directory was found
                if self.logs_base_path.exists():
                    try:
                        all_dirs = [item for item in self.logs_base_path.iterdir() if item.is_dir()]
                        hs_dirs = [item for item in all_dirs if item.name.startswith("Hearthstone_")]
                        self.logger.error(f"  Base path has {len(all_dirs)} directories, {len(hs_dirs)} Hearthstone directories")
                        
                        if hs_dirs:
                            # Show the most recent directories
                            hs_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            for i, dir_path in enumerate(hs_dirs[:3]):
                                age = datetime.now() - datetime.fromtimestamp(dir_path.stat().st_mtime)
                                self.logger.error(f"    #{i+1}: {dir_path.name} ({age.total_seconds()/60:.1f}min ago)")
                    except Exception as e:
                        self.logger.error(f"  Error analyzing base path: {e}")
                
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Recovery failed with exception: {e}")
            return False
    
    def _resolve_hearthstone_logs_path(self, custom_path: Optional[str] = None) -> Path:
        """
        Intelligently resolve Hearthstone logs path across platforms.
        
        Args:
            custom_path: User-specified path (optional)
            
        Returns:
            Path: Resolved logs base path
        """
        if custom_path:
            return Path(custom_path)
        
        # Platform-specific path resolution strategies
        if sys.platform.startswith('win'):
            return self._resolve_windows_logs_path()
        else:
            return self._resolve_linux_wsl_logs_path()
    
    def _resolve_windows_logs_path(self) -> Path:
        """
        Resolve Hearthstone logs path on Windows using multiple strategies.
        
        Returns:
            Path: Windows logs path
        """
        strategies = [
            self._get_hearthstone_install_from_registry,
            self._get_hearthstone_install_from_common_paths,
            self._get_hearthstone_install_from_env_vars,
        ]
        
        for strategy in strategies:
            try:
                install_path = strategy()
                if install_path:
                    logs_path = install_path / "Logs"
                    if logs_path.exists():
                        self.logger.info(f"✅ Found Windows Hearthstone logs: {logs_path}")
                        return logs_path
            except Exception as e:
                self.logger.debug(f"Strategy failed: {strategy.__name__}: {e}")
                continue
        
        # Final fallback for Windows
        fallback_paths = [
            Path("M:/Hearthstone/Logs"),
            Path("C:/Program Files (x86)/Hearthstone/Logs"),
            Path(os.path.expanduser("~/AppData/Local/Blizzard/Hearthstone/Logs")),
        ]
        
        for path in fallback_paths:
            if path.exists():
                self.logger.info(f"✅ Found fallback Windows path: {path}")
                return path
        
        self.logger.warning("⚠️ No Windows Hearthstone logs found, using default")
        return Path("M:/Hearthstone/Logs")
    
    def _resolve_linux_wsl_logs_path(self) -> Path:
        """
        Resolve Hearthstone logs path on Linux/WSL.
        
        Returns:
            Path: Linux/WSL logs path
        """
        wsl_paths = [
            Path("/mnt/m/Hearthstone/Logs"),
            Path("/mnt/c/Program Files (x86)/Hearthstone/Logs"),
            Path("/mnt/d/Games/Hearthstone/Logs"),
        ]
        
        for path in wsl_paths:
            if path.exists():
                self.logger.info(f"✅ Found WSL Hearthstone logs: {path}")
                return path
        
        self.logger.warning("⚠️ No WSL Hearthstone logs found, using default")
        return Path("/mnt/m/Hearthstone/Logs")
    
    def _get_hearthstone_install_from_registry(self) -> Optional[Path]:
        """
        Get Hearthstone installation path from Windows registry.
        
        Returns:
            Optional[Path]: Installation path if found
        """
        if not sys.platform.startswith('win'):
            return None
        
        try:
            import winreg
            
            # Try multiple registry locations
            registry_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Blizzard Entertainment\Hearthstone"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Blizzard Entertainment\Hearthstone"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Blizzard Entertainment\Hearthstone"),
            ]
            
            for hkey, subkey in registry_paths:
                try:
                    with winreg.OpenKey(hkey, subkey) as key:
                        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                        path = Path(install_path)
                        if path.exists():
                            self.logger.info(f"✅ Found Hearthstone via registry: {path}")
                            return path
                except (FileNotFoundError, OSError):
                    continue
            
            return None
            
        except ImportError:
            self.logger.debug("winreg not available (not on Windows)")
            return None
        except Exception as e:
            self.logger.debug(f"Registry lookup failed: {e}")
            return None
    
    def _get_hearthstone_install_from_common_paths(self) -> Optional[Path]:
        """
        Check common Hearthstone installation paths.
        
        Returns:
            Optional[Path]: Installation path if found
        """
        if sys.platform.startswith('win'):
            common_paths = [
                Path("C:/Program Files (x86)/Hearthstone"),
                Path("C:/Program Files/Hearthstone"),
                Path("D:/Games/Hearthstone"),
                Path("E:/Games/Hearthstone"),
            ]
        else:
            common_paths = [
                Path("/mnt/c/Program Files (x86)/Hearthstone"),
                Path("/mnt/d/Games/Hearthstone"),
                Path("/mnt/e/Games/Hearthstone"),
            ]
        
        for path in common_paths:
            if path.exists() and (path / "Hearthstone.exe").exists():
                self.logger.info(f"✅ Found Hearthstone in common path: {path}")
                return path
        
        return None
    
    def _get_hearthstone_install_from_env_vars(self) -> Optional[Path]:
        """
        Check environment variables for Hearthstone path.
        
        Returns:
            Optional[Path]: Installation path if found
        """
        env_vars = ['HEARTHSTONE_PATH', 'HS_PATH', 'HEARTHSTONE_INSTALL']
        
        for var in env_vars:
            path_str = os.environ.get(var)
            if path_str:
                path = Path(path_str)
                if path.exists():
                    self.logger.info(f"✅ Found Hearthstone via {var}: {path}")
                    return path
        
        return None
    
    def find_latest_log_directory(self) -> Optional[Path]:
        """
        ENHANCED: Robust log directory discovery with multi-platform support.
        
        Uses the actual user-provided directory structure as highest priority,
        then falls back to standard detection methods.
        
        Returns:
            Optional[Path]: Path to most recent timestamped log directory, or None if not found
        """
        
        # Priority-ordered base path candidates (user's actual path first)
        base_path_candidates = [
            Path("M:/Hearthstone/Logs"),              # User's confirmed working path
            Path("/mnt/m/Hearthstone/Logs"),          # WSL equivalent of M: drive
            Path("/mnt/c/Program Files (x86)/Hearthstone/Logs"),  # Standard installation
            Path(os.path.expanduser("~/AppData/Local/Blizzard/Hearthstone/Logs")),  # Windows user path
            self.logs_base_path if hasattr(self, 'logs_base_path') else None,  # Current detected path
        ]
        
        # Remove None entries
        base_path_candidates = [p for p in base_path_candidates if p is not None]
        
        self.logger.info(f"🔍 Searching {len(base_path_candidates)} base path candidates...")
        
        for i, candidate_path in enumerate(base_path_candidates):
            self.logger.debug(f"  Candidate #{i+1}: {candidate_path}")
            
            try:
                if not candidate_path.exists():
                    self.logger.debug(f"    ❌ Path does not exist")
                    continue
                    
                if not candidate_path.is_dir():
                    self.logger.debug(f"    ❌ Path is not a directory")
                    continue
                
                self.logger.debug(f"    ✅ Path exists and is accessible")
                
                # Scan for timestamped Hearthstone directories
                timestamped_dirs = self._scan_timestamped_directories(candidate_path)
                
                if timestamped_dirs:
                    self.logger.info(f"✅ Found {len(timestamped_dirs)} timestamped directories in: {candidate_path}")
                    
                    # Update our base path to the successful candidate
                    self.logs_base_path = candidate_path
                    
                    # Return the most recent directory
                    most_recent = self._select_most_recent_directory(timestamped_dirs)
                    self.logger.info(f"📂 Selected most recent: {most_recent.name}")
                    return most_recent
                else:
                    self.logger.debug(f"    ⚠️ No timestamped directories found")
                    
            except (OSError, PermissionError) as e:
                self.logger.debug(f"    ❌ Access error: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"    ❌ Unexpected error: {e}")
                continue
        
        # All candidates failed
        self.logger.error("❌ No valid Hearthstone log directories found in any candidate path")
        self._diagnose_discovery_failure(base_path_candidates)
        return None
    
    def _scan_timestamped_directories(self, base_path: Path) -> List[Tuple[datetime, Path]]:
        """
        Scan a base directory for Hearthstone timestamped subdirectories.
        
        Args:
            base_path: The directory to scan
            
        Returns:
            List of (timestamp, path) tuples for valid directories
        """
        timestamped_dirs = []
        
        try:
            for item in base_path.iterdir():
                if not item.is_dir():
                    continue
                    
                # Check if directory name matches Hearthstone pattern
                if not item.name.startswith("Hearthstone_"):
                    continue
                
                # Parse timestamp from directory name: Hearthstone_YYYY_MM_DD_HH_MM_SS
                try:
                    parts = item.name.split("_")
                    if len(parts) >= 6:
                        year, month, day, hour, minute, second = parts[1:7]
                        timestamp = datetime(
                            int(year), int(month), int(day),
                            int(hour), int(minute), int(second)
                        )
                        timestamped_dirs.append((timestamp, item))
                        self.logger.debug(f"    📁 Found valid directory: {item.name} ({timestamp})")
                    else:
                        self.logger.debug(f"    ⚠️ Invalid directory name format: {item.name}")
                except (ValueError, IndexError) as e:
                    self.logger.debug(f"    ⚠️ Failed to parse timestamp from {item.name}: {e}")
                    continue
                    
        except (OSError, PermissionError) as e:
            self.logger.error(f"❌ Cannot scan directory {base_path}: {e}")
            return []
        
        return timestamped_dirs
    
    def _select_most_recent_directory(self, timestamped_dirs: List[Tuple[datetime, Path]]) -> Path:
        """
        Select the most recent directory from a list of timestamped directories.
        
        Args:
            timestamped_dirs: List of (timestamp, path) tuples
            
        Returns:
            Path: The most recent directory
        """
        # Sort by timestamp (most recent first)
        timestamped_dirs.sort(key=lambda x: x[0], reverse=True)
        
        most_recent_timestamp, most_recent_path = timestamped_dirs[0]
        
        # Check if the most recent directory is reasonably fresh
        age = datetime.now() - most_recent_timestamp
        if age > timedelta(hours=24):
            self.logger.warning(f"⚠️ Most recent log directory is {age.total_seconds()/3600:.1f} hours old")
        else:
            self.logger.info(f"✅ Most recent directory is {age.total_seconds()/60:.1f} minutes old")
        
        return most_recent_path
    
    def _diagnose_discovery_failure(self, attempted_paths: List[Path]):
        """
        Provide detailed diagnostics when directory discovery fails.
        
        Args:
            attempted_paths: List of paths that were attempted
        """
        self.logger.error("🔍 DIRECTORY DISCOVERY FAILURE DIAGNOSTICS:")
        
        for i, path in enumerate(attempted_paths):
            self.logger.error(f"  Candidate #{i+1}: {path}")
            
            if path.exists():
                self.logger.error(f"    Status: EXISTS")
                try:
                    if path.is_dir():
                        contents = list(path.iterdir())
                        dirs = [item for item in contents if item.is_dir()]
                        hs_dirs = [item for item in dirs if item.name.startswith("Hearthstone_")]
                        
                        self.logger.error(f"    Contents: {len(contents)} items, {len(dirs)} directories")
                        self.logger.error(f"    Hearthstone dirs: {len(hs_dirs)}")
                        
                        if hs_dirs:
                            # Show first few directories
                            for j, hs_dir in enumerate(hs_dirs[:3]):
                                age_seconds = (datetime.now() - datetime.fromtimestamp(hs_dir.stat().st_mtime)).total_seconds()
                                self.logger.error(f"      #{j+1}: {hs_dir.name} ({age_seconds/60:.1f}min ago)")
                    else:
                        self.logger.error(f"    Status: EXISTS but is not a directory")
                except Exception as e:
                    self.logger.error(f"    Status: EXISTS but cannot read contents: {e}")
            else:
                self.logger.error(f"    Status: DOES NOT EXIST")
    
    def discover_log_files(self, log_dir: Path) -> Dict[str, Path]:
        """
        Discover available log files in the directory.
        Returns mapping of log type to file path.
        """
        log_files = {}
        
        # Arena Tracker critical log files
        important_logs = {
            'Arena.log': 'arena',
            'Asset.log': 'asset', 
            'LoadingScreen.log': 'loading',
            'Hearthstone.log': 'game',  # Sometimes contains Power.log content
            'BattleNet.log': 'battlenet'
        }
        
        for log_file, log_type in important_logs.items():
            log_path = log_dir / log_file
            if log_path.exists():
                log_files[log_type] = log_path
                self.logger.info(f"✅ Found {log_type}: {log_file}")
            else:
                self.logger.warning(f"⚠️ Missing {log_type}: {log_file}")
        
        # Look for any additional .log files
        for log_path in log_dir.glob("*.log"):
            log_name = log_path.name.lower()
            if 'power' in log_name and 'power' not in log_files:
                log_files['power'] = log_path
                self.logger.info(f"✅ Found power log: {log_path.name}")
            elif 'zone' in log_name and 'zone' not in log_files:
                log_files['zone'] = log_path
                self.logger.info(f"✅ Found zone log: {log_path.name}")
        
        return log_files
    
    def read_new_log_lines(self, log_path: Path, log_type: str) -> List[str]:
        """
        Read new lines from a log file since last read.
        Uses Arena Tracker's incremental reading approach.
        """
        try:
            if not log_path.exists():
                return []
            
            current_size = log_path.stat().st_size
            last_position = self.log_positions.get(log_type, 0)
            
            if current_size <= last_position:
                return []  # No new content
            
            # Try different encoding methods for Windows files
            encodings_to_try = ['utf-8', 'utf-16', 'cp1252', 'latin1']
            
            for encoding in encodings_to_try:
                try:
                    with open(log_path, 'r', encoding=encoding, errors='ignore') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        self.log_positions[log_type] = f.tell()
                    
                    return [line.strip() for line in new_lines if line.strip()]
                    
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as inner_e:
                    if encoding == encodings_to_try[-1]:  # Last encoding attempt
                        raise inner_e
                    continue
            
            return []
            
        except (OSError, PermissionError) as e:
            self.logger.error(f"❌ Access error reading {log_type} ({log_path}): {e}")
            return []
        except Exception as e:
            self.logger.error(f"❌ Error reading {log_type}: {e}")
            return []
    
    def parse_arena_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse Arena.log line using Arena Tracker patterns."""
        try:
            # Extract timestamp and message
            if line.startswith('D '):
                parts = line.split(' ', 2)
                if len(parts) >= 3:
                    timestamp_str = parts[1]
                    message = parts[2]
                    
                    # Parse timestamp (format: HH:MM:SS.ffffff)
                    try:
                        time_part = timestamp_str.split('.')[0]
                        hour, minute, second = map(int, time_part.split(':'))
                        
                        # Use today's date with the parsed time
                        today = datetime.now().replace(hour=hour, minute=minute, second=second, microsecond=0)
                        
                        return LogEntry(
                            timestamp=today,
                            component="Arena",
                            message=message,
                            raw_line=line
                        )
                    except ValueError:
                        pass
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing arena log line: {e}")
            return None
    
    def process_arena_events(self, lines: List[str]):
        """Process Arena.log events using Arena Tracker methodology."""
        for line in lines:
            entry = self.parse_arena_log_line(line)
            if not entry:
                continue
            
            message = entry.message
            
            # Draft pick detection (Arena Tracker pattern)
            pick_match = self.patterns['draft_pick'].search(message)
            if pick_match:
                slot = int(pick_match.group(1))
                card_code = pick_match.group(2)
                is_premium = pick_match.group(3) == "True"
                
                pick = DraftPick(
                    slot=slot,
                    card_code=card_code,
                    is_premium=is_premium,
                    timestamp=entry.timestamp
                )
                
                self.current_draft_picks.append(pick)
                
                self.logger.info(f"🎯 DRAFT PICK: Slot {slot} -> {card_code} {'✨' if is_premium else ''}")
                
                if self.on_draft_pick:
                    self.on_draft_pick(pick)
                
                # Check if draft is complete (30 picks)
                if len(self.current_draft_picks) >= 30:
                    self._set_game_state(GameState.ARENA_DRAFT)
                    if self.on_draft_complete:
                        self.on_draft_complete(self.current_draft_picks)
            
            # Hero selection
            hero_match = self.patterns['draft_hero'].search(message)
            if hero_match:
                hero_code = hero_match.group(1)
                self.current_hero = hero_code
                self.logger.info(f"👑 HERO SELECTED: {hero_code}")
            
            # Draft start detection (Enhanced for AI Helper integration)
            if self.patterns['draft_choices'].search(message):
                if self.current_game_state != GameState.ARENA_DRAFT:
                    self._display_draft_start()
                    self.current_draft_picks.clear()
                    self._set_game_state(GameState.ARENA_DRAFT)
                    if self.on_draft_start:
                        self.on_draft_start()
                        
                # NEW: Also check for detailed draft choices pattern
                detailed_match = self.patterns['draft_choices_detailed'].search(message)
                if detailed_match:
                    self.logger.info("🎯 DETAILED DRAFT CHOICES DETECTED - AI Helper integration ready")
                    # This enhanced pattern detection can be used by AI Helper for more accurate timing
            
            # Current deck contents (for mid-draft analysis)
            deck_card_match = self.patterns['draft_deck_card'].search(message)
            if deck_card_match:
                card_code = deck_card_match.group(1)
                self.logger.info(f"📋 Current deck contains: {card_code}")
    
    def process_loading_screen_events(self, lines: List[str]):
        """Process LoadingScreen.log events for precise screen detection."""
        for line in lines:
            # Scene loaded detection (current screen)
            scene_loaded_match = self.patterns['scene_loaded'].search(line)
            if scene_loaded_match:
                scene_mode = scene_loaded_match.group(1).upper()
                self._map_scene_to_game_state(scene_mode, "LOADED")
            
            # Scene unload detection (transitioning to next screen)
            scene_unload_match = self.patterns['scene_unload'].search(line)
            if scene_unload_match:
                next_mode = scene_unload_match.group(1).upper()
                self._map_scene_to_game_state(next_mode, "LOADING")
    
    def _map_scene_to_game_state(self, scene_mode: str, action: str):
        """Map Hearthstone scene modes to game states with prominent display."""
        scene_mapping = {
            'LOGIN': GameState.LOGIN,
            'HUB': GameState.HUB,
            'DRAFT': GameState.ARENA_DRAFT,
            'GAMEPLAY': GameState.GAMEPLAY,
            'COLLECTION': GameState.COLLECTION,
            'TOURNAMENT': GameState.TOURNAMENT,
            'BATTLEGROUNDS': GameState.BATTLEGROUNDS,
            'ADVENTURE': GameState.ADVENTURE,
            'TAVERN_BRAWL': GameState.TAVERN_BRAWL,
            'SHOP': GameState.SHOP,
        }
        
        new_state = scene_mapping.get(scene_mode, GameState.UNKNOWN)
        
        if action == "LOADED":
            # Show prominent screen detection
            self._display_prominent_screen_change(new_state)
            self._set_game_state(new_state)
        elif action == "LOADING":
            self.logger.info(f"🔄 Transitioning to: {new_state.value}...")
    
    def _display_prominent_screen_change(self, new_state: GameState):
        """Display very prominent screen change notification."""
        # Create a prominent visual separator
        self.logger.info("\n" + "#" * 80)
        self.logger.info("#" + " " * 78 + "#")
        
        # Map states to emojis and colors
        state_display = {
            GameState.LOGIN: ("🔐", "LOGIN SCREEN"),
            GameState.HUB: ("🏠", "MAIN MENU"),
            GameState.ARENA_DRAFT: ("🎯", "ARENA DRAFT"),
            GameState.DRAFT_COMPLETE: ("✅", "DRAFT COMPLETE"),
            GameState.GAMEPLAY: ("⚔️", "PLAYING MATCH"),
            GameState.COLLECTION: ("📚", "COLLECTION"),
            GameState.TOURNAMENT: ("🏆", "TOURNAMENT"),
            GameState.BATTLEGROUNDS: ("🥊", "BATTLEGROUNDS"),
            GameState.ADVENTURE: ("🗺️", "ADVENTURE"),
            GameState.TAVERN_BRAWL: ("🍺", "TAVERN BRAWL"),
            GameState.SHOP: ("🛒", "SHOP"),
            GameState.UNKNOWN: ("❓", "UNKNOWN SCREEN"),
        }
        
        emoji, display_name = state_display.get(new_state, ("❓", "UNKNOWN"))
        
        # Center the text
        message = f"{emoji} CURRENT SCREEN: {display_name} {emoji}"
        padding = (78 - len(message)) // 2
        self.logger.info("#" + " " * padding + message + " " * (78 - padding - len(message)) + "#")
        
        self.logger.info("#" + " " * 78 + "#")
        self.logger.info("#" * 80)
        
        # Add context-specific information
        if new_state == GameState.HUB:
            print("🎮 You're in the main menu - ready to start Arena!")
        elif new_state == GameState.ARENA_DRAFT:
            print("🎯 Arena draft detected - monitoring for card picks...")
        elif new_state == GameState.GAMEPLAY:
            print("⚔️ Match in progress - Arena Bot on standby")
        elif new_state == GameState.COLLECTION:
            print("📚 Collection browsing - Arena Bot waiting")
        
        self.logger.info("")
    
    def _display_draft_start(self):
        """Display prominent draft start notification with platform-safe characters."""
        # Use platform-safe characters
        icon = "*"  # Always use ASCII for maximum compatibility
        
        self.logger.info("\n" + icon * 40)
        self.logger.info(icon + " " * 76 + icon)
        self.logger.info(icon + " " * 25 + "ARENA DRAFT STARTED!" + " " * 25 + icon)
        self.logger.info(icon + " " * 20 + "Monitoring for card picks..." + " " * 20 + icon)
        self.logger.info(icon + " " * 76 + icon)
        self.logger.info(icon * 40)
        self.logger.info("")
    
    def _set_game_state(self, new_state: GameState):
        """Update game state and notify listeners."""
        if new_state != self.current_game_state:
            old_state = self.current_game_state
            self.current_game_state = new_state
            
            self.logger.info(f"🎮 Game state changed: {old_state.value} -> {new_state.value}")
            
            if self.on_game_state_change:
                self.on_game_state_change(old_state, new_state)
    
    def monitoring_loop(self):
        """
        Main monitoring loop - Arena Tracker style with AI Helper enhancements.
        Enhanced with heartbeat monitoring, event deduplication, and error recovery.
        """
        self.logger.info("🚀 Starting enhanced log monitoring loop with AI Helper integration...")
        
        while self.monitoring:
            try:
                # NEW: Check heartbeat and log accessibility
                if not self._check_heartbeat_and_log_accessibility():
                    self.logger.warning("⚠️ Log files not accessible, attempting recovery...")
                    time.sleep(5)
                    continue
                
                # Check if we need to find a new log directory
                if not self.current_log_dir:
                    self.current_log_dir = self.find_latest_log_directory()
                    if self.current_log_dir:
                        self.log_files = self.discover_log_files(self.current_log_dir)
                        self.log_positions = {}  # Reset positions for new directory
                    else:
                        self.logger.warning("⚠️ Cannot find log directory, retrying in 10 seconds...")
                        time.sleep(10)
                        continue
                
                if not self.log_files:
                    self.logger.warning("⚠️ No log files found, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Process each log file with enhanced error handling
                for log_type, log_path in self.log_files.items():
                    try:
                        new_lines = self.read_new_log_lines(log_path, log_type)
                        
                        if new_lines:
                            # Filter out duplicate events
                            filtered_lines = []
                            for line in new_lines:
                                event_signature = self._generate_event_signature(line.message, log_type)
                                if not self._is_duplicate_event(event_signature):
                                    filtered_lines.append(line)
                            
                            if filtered_lines:
                                self.logger.info(f"📖 Processing {len(filtered_lines)} new lines from {log_type} (filtered {len(new_lines) - len(filtered_lines)} duplicates)")
                                
                                if log_type == 'arena':
                                    self.process_arena_events(filtered_lines)
                                elif log_type == 'loading':
                                    self.process_loading_screen_events(filtered_lines)
                                # Add more log processors as needed
                            
                    except Exception as log_error:
                        self.logger.warning(f"⚠️ Error processing {log_type} log: {log_error}")
                        # Continue with other log files instead of crashing
                        continue
                
                # Adaptive sleep - more frequent when in draft
                sleep_time = 0.5 if self.current_game_state == GameState.ARENA_DRAFT else 2.0
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                # Attempt error recovery
                self._attempt_log_error_recovery()
                time.sleep(5)  # Wait before retrying
    
    def start_monitoring(self):
        """Start the log monitoring system."""
        if self.monitoring:
            self.logger.warning("⚠️ Already monitoring")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("✅ Log monitoring started")
    
    def stop_monitoring(self):
        """Stop the log monitoring system."""
        self.monitoring = False
        self.logger.info("⏸️ Log monitoring stopped")
    
    def get_current_state(self) -> Dict:
        """Get current game state information."""
        return {
            'game_state': self.current_game_state.value,
            'log_directory': str(self.current_log_dir) if self.current_log_dir else None,
            'available_logs': list(self.log_files.keys()),
            'draft_picks_count': len(self.current_draft_picks),
            'current_hero': self.current_hero,
            'recent_picks': [
                {
                    'slot': pick.slot,
                    'card_code': pick.card_code,
                    'is_premium': pick.is_premium,
                    'timestamp': pick.timestamp.isoformat()
                }
                for pick in self.current_draft_picks[-5:]  # Last 5 picks
            ]
        }

def main():
    """Demo of the log monitoring system."""
    print("🎯 HEARTHSTONE LOG MONITOR - ARENA TRACKER STYLE")
    print("=" * 60)
    
    monitor = HearthstoneLogMonitor()
    
    # Set up callbacks
    def on_state_change(old_state, new_state):
        print(f"\n🎮 STATE CHANGE: {old_state.value} -> {new_state.value}")
    
    def on_draft_pick(pick):
        print(f"\n🎯 DRAFT PICK #{len(monitor.current_draft_picks)}: {pick.card_code}")
        if pick.is_premium:
            print("   ✨ GOLDEN CARD!")
    
    def on_draft_start():
        print(f"\n🎮 ARENA DRAFT STARTED!")
        print("   Waiting for card picks...")
    
    monitor.on_game_state_change = on_state_change
    monitor.on_draft_pick = on_draft_pick  
    monitor.on_draft_start = on_draft_start
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        print("\n🔍 Monitoring Hearthstone logs...")
        print("📝 Current state:")
        state = monitor.get_current_state()
        for key, value in state.items():
            print(f"   {key}: {value}")
        
        print("\n⏸️  Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏸️ Stopping monitor...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()