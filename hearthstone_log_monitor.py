#!/usr/bin/env python3
"""
Hearthstone Log Monitor - Arena Tracker Style
Monitors Hearthstone log files for real-time game state detection.
Based on Arena Tracker's proven log monitoring methodology.
"""

import os
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
from enum import Enum

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
    
    def __init__(self, logs_base_path: str = "/mnt/m/Hearthstone/Logs"):
        """Initialize the log monitor."""
        self.logs_base_path = Path(logs_base_path)
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
        
        # Arena Tracker-style regex patterns
        self.patterns = {
            'draft_pick': re.compile(r'DraftManager\.OnChosen.*Slot=(\d+).*cardId=([A-Z0-9_]+).*Premium=(\w+)'),
            'draft_hero': re.compile(r'DraftManager\.OnHeroChosen.*HeroCardID=([A-Z0-9_]+)'),
            'draft_choices': re.compile(r'DraftManager\.OnChoicesAndContents'),
            'draft_deck_card': re.compile(r'Draft deck contains card ([A-Z0-9_]+)'),
            'scene_loaded': re.compile(r'LoadingScreen\.OnSceneLoaded.*currMode=(\w+)'),
            'scene_unload': re.compile(r'LoadingScreen\.OnScenePreUnload.*nextMode=(\w+)'),
            'asset_load': re.compile(r'AssetLoader.*Loading.*([A-Z0-9_]+)'),
        }
        
        print("🎯 Hearthstone Log Monitor Initialized")
        print(f"📁 Monitoring: {self.logs_base_path}")
    
    def find_latest_log_directory(self) -> Optional[Path]:
        """
        Find the most recent Hearthstone log directory.
        Directories are named like: Hearthstone_2025_07_11_12_15_01
        """
        # Try different path formats for WSL compatibility
        possible_paths = [
            self.logs_base_path,
            Path("/mnt/m/Hearthstone/Logs"),
            Path("M:/Hearthstone/Logs"),
        ]
        
        working_path = None
        for path_to_try in possible_paths:
            try:
                if path_to_try.exists():
                    working_path = path_to_try
                    self.logs_base_path = working_path
                    print(f"✅ Found working log path: {working_path}")
                    break
            except (OSError, PermissionError) as e:
                print(f"⚠️ Cannot access {path_to_try}: {e}")
                continue
        
        if not working_path:
            print(f"❌ No accessible log directory found")
            return None
        
        # Find all Hearthstone log directories
        log_dirs = []
        try:
            for item in self.logs_base_path.iterdir():
                if item.is_dir() and item.name.startswith("Hearthstone_"):
                    try:
                        # Extract timestamp from directory name
                        parts = item.name.split("_")
                        if len(parts) >= 6:
                            year, month, day, hour, minute, second = parts[1:7]
                            timestamp = datetime(
                                int(year), int(month), int(day),
                                int(hour), int(minute), int(second)
                            )
                            log_dirs.append((timestamp, item))
                    except (ValueError, IndexError):
                        continue
        except (OSError, PermissionError) as e:
            print(f"❌ Cannot iterate log directories: {e}")
            return None
        
        if not log_dirs:
            print("❌ No valid Hearthstone log directories found")
            return None
        
        # Sort by timestamp and get the most recent
        log_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_timestamp, latest_dir = log_dirs[0]
        
        # Check if the latest directory is recent (within last 24 hours)
        if datetime.now() - latest_timestamp > timedelta(hours=24):
            print(f"⚠️ Latest log directory is old: {latest_timestamp}")
        
        print(f"📂 Found latest log directory: {latest_dir.name}")
        print(f"🕒 Timestamp: {latest_timestamp}")
        
        return latest_dir
    
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
                print(f"✅ Found {log_type}: {log_file}")
            else:
                print(f"⚠️ Missing {log_type}: {log_file}")
        
        # Look for any additional .log files
        for log_path in log_dir.glob("*.log"):
            log_name = log_path.name.lower()
            if 'power' in log_name and 'power' not in log_files:
                log_files['power'] = log_path
                print(f"✅ Found power log: {log_path.name}")
            elif 'zone' in log_name and 'zone' not in log_files:
                log_files['zone'] = log_path
                print(f"✅ Found zone log: {log_path.name}")
        
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
            print(f"❌ Access error reading {log_type} ({log_path}): {e}")
            return []
        except Exception as e:
            print(f"❌ Error reading {log_type}: {e}")
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
            print(f"❌ Error parsing arena log line: {e}")
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
                
                print(f"🎯 DRAFT PICK: Slot {slot} -> {card_code} {'✨' if is_premium else ''}")
                
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
                print(f"👑 HERO SELECTED: {hero_code}")
            
            # Draft start detection
            if self.patterns['draft_choices'].search(message):
                if self.current_game_state != GameState.ARENA_DRAFT:
                    self._display_draft_start()
                    self.current_draft_picks.clear()
                    self._set_game_state(GameState.ARENA_DRAFT)
                    if self.on_draft_start:
                        self.on_draft_start()
            
            # Current deck contents (for mid-draft analysis)
            deck_card_match = self.patterns['draft_deck_card'].search(message)
            if deck_card_match:
                card_code = deck_card_match.group(1)
                print(f"📋 Current deck contains: {card_code}")
    
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
            print(f"🔄 Transitioning to: {new_state.value}...")
    
    def _display_prominent_screen_change(self, new_state: GameState):
        """Display very prominent screen change notification."""
        # Create a prominent visual separator
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        
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
        print("█" + " " * padding + message + " " * (78 - padding - len(message)) + "█")
        
        print("█" + " " * 78 + "█")
        print("█" * 80)
        
        # Add context-specific information
        if new_state == GameState.HUB:
            print("🎮 You're in the main menu - ready to start Arena!")
        elif new_state == GameState.ARENA_DRAFT:
            print("🎯 Arena draft detected - monitoring for card picks...")
        elif new_state == GameState.GAMEPLAY:
            print("⚔️ Match in progress - Arena Bot on standby")
        elif new_state == GameState.COLLECTION:
            print("📚 Collection browsing - Arena Bot waiting")
        
        print()
    
    def _display_draft_start(self):
        """Display prominent draft start notification."""
        print("\n" + "🎯" * 40)
        print("🎯" + " " * 76 + "🎯")
        print("🎯" + " " * 25 + "ARENA DRAFT STARTED!" + " " * 25 + "🎯")
        print("🎯" + " " * 20 + "Monitoring for card picks..." + " " * 20 + "🎯")
        print("🎯" + " " * 76 + "🎯")
        print("🎯" * 40)
        print()
    
    def _set_game_state(self, new_state: GameState):
        """Update game state and notify listeners."""
        if new_state != self.current_game_state:
            old_state = self.current_game_state
            self.current_game_state = new_state
            
            print(f"🎮 Game state changed: {old_state.value} -> {new_state.value}")
            
            if self.on_game_state_change:
                self.on_game_state_change(old_state, new_state)
    
    def monitoring_loop(self):
        """Main monitoring loop - Arena Tracker style."""
        print("🚀 Starting log monitoring loop...")
        
        while self.monitoring:
            try:
                # Check if we need to find a new log directory
                if not self.current_log_dir:
                    self.current_log_dir = self.find_latest_log_directory()
                    if self.current_log_dir:
                        self.log_files = self.discover_log_files(self.current_log_dir)
                        self.log_positions = {}  # Reset positions for new directory
                    else:
                        print("⚠️ Cannot find log directory, retrying in 10 seconds...")
                        time.sleep(10)
                        continue
                
                if not self.log_files:
                    print("⚠️ No log files found, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Process each log file
                for log_type, log_path in self.log_files.items():
                    new_lines = self.read_new_log_lines(log_path, log_type)
                    
                    if new_lines:
                        print(f"📖 Processing {len(new_lines)} new lines from {log_type}")
                        
                        if log_type == 'arena':
                            self.process_arena_events(new_lines)
                        elif log_type == 'loading':
                            self.process_loading_screen_events(new_lines)
                        # Add more log processors as needed
                
                # Adaptive sleep - more frequent when in draft
                sleep_time = 0.5 if self.current_game_state == GameState.ARENA_DRAFT else 2.0
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """Start the log monitoring system."""
        if self.monitoring:
            print("⚠️ Already monitoring")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ Log monitoring started")
    
    def stop_monitoring(self):
        """Stop the log monitoring system."""
        self.monitoring = False
        print("⏸️ Log monitoring stopped")
    
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