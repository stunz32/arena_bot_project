#!/usr/bin/env python3
"""
Debug Image Manager
Organizes and manages debug images with automatic cleanup and size limits.
"""

import os
import cv2
import time
import shutil
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta


class DebugImageManager:
    """Manages debug images with organization and cleanup."""
    
    def __init__(self, base_dir: str = "/home/marcco/arena_bot_project"):
        self.base_dir = Path(base_dir)
        self.debug_dir = self.base_dir / "debug_images"
        
        # Directory structure
        self.dirs = {
            'detection': self.debug_dir / "detection",     # Card detection results
            'cards': self.debug_dir / "cards",             # Extracted card images  
            'interface': self.debug_dir / "interface",     # Interface detection
            'archive': self.debug_dir / "archive"          # Old images before cleanup
        }
        
        # Cleanup settings
        self.max_images_per_dir = 50      # Max images per category
        self.max_age_hours = 24           # Max age before cleanup
        self.max_total_size_mb = 100      # Max total size in MB
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create debug directories if they don't exist."""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_debug_image(self, image, filename: str, category: str = 'detection', 
                        timestamp: bool = True) -> str:
        """
        Save a debug image in organized directory structure.
        
        Args:
            image: OpenCV image array
            filename: Base filename (without extension)
            category: 'detection', 'cards', 'interface', or 'archive'
            timestamp: Whether to add timestamp to filename
            
        Returns:
            Full path to saved image
        """
        # Clean up old images first
        self._cleanup_old_images()
        
        # Prepare filename
        if timestamp:
            timestamp_str = datetime.now().strftime("%H%M%S")
            filename = f"{timestamp_str}_{filename}"
        
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Get target directory
        target_dir = self.dirs.get(category, self.dirs['detection'])
        
        # Save image
        file_path = target_dir / filename
        cv2.imwrite(str(file_path), image)
        
        return str(file_path)
    
    def save_detection_results(self, screenshot, results: Dict, session_id: str = None):
        """Save complete detection session results."""
        if session_id is None:
            session_id = datetime.now().strftime("%H%M%S")
        
        # Save original screenshot
        self.save_debug_image(
            screenshot, 
            f"session_{session_id}_screenshot", 
            'detection', 
            timestamp=False
        )
        
        # Save extracted cards if available
        if 'detected_cards' in results:
            for i, card in enumerate(results['detected_cards'], 1):
                coords = card.get('coordinates')
                if coords:
                    x, y, w, h = coords
                    card_image = screenshot[y:y+h, x:x+w]
                    
                    card_name = card.get('card_name', 'unknown').replace(' ', '_')
                    self.save_debug_image(
                        card_image,
                        f"session_{session_id}_card_{i}_{card_name}",
                        'cards',
                        timestamp=False
                    )
    
    def _cleanup_old_images(self):
        """Clean up old debug images based on limits."""
        for category, dir_path in self.dirs.items():
            if category == 'archive':  # Don't auto-cleanup archive
                continue
                
            self._cleanup_directory(dir_path, category)
    
    def _cleanup_directory(self, dir_path: Path, category: str):
        """Clean up a specific directory."""
        if not dir_path.exists():
            return
        
        # Get all PNG files with their stats
        files = []
        for file_path in dir_path.glob("*.png"):
            try:
                stat = file_path.stat()
                files.append({
                    'path': file_path,
                    'mtime': stat.st_mtime,
                    'size': stat.st_size
                })
            except OSError:
                continue
        
        if not files:
            return
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Remove by count limit
        if len(files) > self.max_images_per_dir:
            excess_files = files[self.max_images_per_dir:]
            self._archive_files([f['path'] for f in excess_files])
        
        # Remove by age limit
        cutoff_time = time.time() - (self.max_age_hours * 3600)
        old_files = [f for f in files if f['mtime'] < cutoff_time]
        if old_files:
            self._archive_files([f['path'] for f in old_files])
    
    def _archive_files(self, file_paths: List[Path]):
        """Move files to archive directory."""
        archive_dir = self.dirs['archive']
        
        for file_path in file_paths:
            try:
                if file_path.exists():
                    # Create unique archive filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_name = f"{timestamp}_{file_path.name}"
                    archive_path = archive_dir / archive_name
                    
                    # Move to archive
                    shutil.move(str(file_path), str(archive_path))
                    
            except Exception as e:
                print(f"Warning: Could not archive {file_path}: {e}")
    
    def cleanup_project_directory(self):
        """Clean up scattered debug images in main project directory."""
        print("🧹 Cleaning up scattered debug images...")
        
        moved_count = 0
        for file_path in self.base_dir.glob("*.png"):
            # Skip if already in debug_images directory
            if 'debug_images' in str(file_path):
                continue
            
            try:
                # Determine category from filename
                filename = file_path.name.lower()
                if any(keyword in filename for keyword in ['card', 'extracted']):
                    category = 'cards'
                elif any(keyword in filename for keyword in ['interface', 'red_area']):
                    category = 'interface'
                else:
                    category = 'archive'  # Archive old debug images
                
                # Move to appropriate directory
                target_dir = self.dirs[category]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"cleanup_{timestamp}_{file_path.name}"
                target_path = target_dir / new_name
                
                shutil.move(str(file_path), str(target_path))
                moved_count += 1
                
            except Exception as e:
                print(f"Warning: Could not move {file_path}: {e}")
        
        print(f"✅ Moved {moved_count} scattered debug images to organized directories")
        return moved_count
    
    def get_stats(self) -> Dict:
        """Get debug image statistics."""
        stats = {}
        total_size = 0
        total_count = 0
        
        for category, dir_path in self.dirs.items():
            if not dir_path.exists():
                stats[category] = {'count': 0, 'size_mb': 0}
                continue
            
            files = list(dir_path.glob("*.png"))
            size_bytes = sum(f.stat().st_size for f in files if f.exists())
            size_mb = size_bytes / (1024 * 1024)
            
            stats[category] = {
                'count': len(files),
                'size_mb': round(size_mb, 2)
            }
            
            total_size += size_mb
            total_count += len(files)
        
        stats['total'] = {
            'count': total_count,
            'size_mb': round(total_size, 2)
        }
        
        return stats
    
    def force_cleanup(self, max_files_per_category: int = 10):
        """Force aggressive cleanup to reduce image count."""
        print(f"🧹 Force cleanup: keeping max {max_files_per_category} files per category")
        
        for category, dir_path in self.dirs.items():
            if not dir_path.exists() or category == 'archive':
                continue
            
            files = list(dir_path.glob("*.png"))
            if len(files) <= max_files_per_category:
                continue
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove excess files
            excess_files = files[max_files_per_category:]
            for file_path in excess_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
            
            print(f"  {category}: removed {len(excess_files)} files")


def main():
    """Test and demonstrate the debug image manager."""
    print("🖼️  DEBUG IMAGE MANAGER")
    print("=" * 50)
    
    manager = DebugImageManager()
    
    # Show current stats
    stats = manager.get_stats()
    print("\n📊 Current debug image statistics:")
    for category, data in stats.items():
        print(f"  {category}: {data['count']} files, {data['size_mb']} MB")
    
    # Clean up scattered files
    print(f"\n🧹 Total images in project: {len(list(Path('/home/marcco/arena_bot_project').glob('*.png')))}")
    moved = manager.cleanup_project_directory()
    
    # Show updated stats
    stats = manager.get_stats()
    print("\n📊 After cleanup:")
    for category, data in stats.items():
        print(f"  {category}: {data['count']} files, {data['size_mb']} MB")
    
    # Force cleanup if still too many
    if stats['total']['count'] > 200:
        print("\n🚨 Still too many debug images, forcing aggressive cleanup...")
        manager.force_cleanup(max_files_per_category=20)
        
        stats = manager.get_stats()
        print("\n📊 After force cleanup:")
        for category, data in stats.items():
            print(f"  {category}: {data['count']} files, {data['size_mb']} MB")
    
    print("\n✅ Debug image management complete!")


if __name__ == "__main__":
    main()