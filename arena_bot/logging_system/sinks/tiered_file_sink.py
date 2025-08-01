"""
Tiered File Sink for S-Tier Logging System.

This module provides intelligent file-based logging with automatic rotation,
compression, tiered storage, and advanced file management capabilities.

Features:
- Multi-tier storage strategy (current, recent, archived)
- Automatic file rotation by size, time, or count
- Compression of archived files (gzip, lz4, zstd)
- Intelligent cleanup policies
- Lock-free file operations where possible
- File integrity verification
- Atomic file operations
- Performance monitoring and optimization
"""

import os
import time
import threading
import logging
import gzip
import shutil
import hashlib
import tempfile
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque

# Import compression libraries with fallbacks
try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# Import from our components
from .base_sink import BaseSink, SinkState, ErrorHandlingStrategy
from ..formatters.structured_formatter import StructuredFormatter
from ..core.hybrid_async_queue import LogMessage


class CompressionType(Enum):
    """File compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


class RotationStrategy(Enum):
    """File rotation strategies."""
    SIZE = "size"          # Rotate by file size
    TIME = "time"          # Rotate by time interval
    DAILY = "daily"        # Rotate daily
    HOURLY = "hourly"      # Rotate hourly
    COUNT = "count"        # Rotate by message count


class StorageTier(Enum):
    """Storage tier levels."""
    CURRENT = "current"    # Active log file
    RECENT = "recent"      # Recent rotated files
    ARCHIVED = "archived"  # Compressed archived files


@dataclass
class TierConfig:
    """Configuration for a storage tier."""
    max_files: int = 10
    max_age_days: int = 30
    compression: CompressionType = CompressionType.GZIP
    compression_level: int = 6
    enable_verification: bool = True


@dataclass
class RotationConfig:
    """File rotation configuration."""
    strategy: RotationStrategy = RotationStrategy.SIZE
    max_size_mb: float = 100.0
    rotation_interval_hours: int = 24
    max_files_per_tier: int = 10
    compression_delay_minutes: int = 5


class FileManager:
    """
    Thread-safe file manager with atomic operations.
    
    Handles file creation, rotation, compression, and cleanup with
    proper error handling and recovery mechanisms.
    """
    
    def __init__(self, base_path: Path, name: str):
        self.base_path = Path(base_path)
        self.name = name
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.FileManager")
        
        # Create directory structure
        self.current_dir = self.base_path / "current"
        self.recent_dir = self.base_path / "recent"
        self.archived_dir = self.base_path / "archived"
        
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create directory structure."""
        try:
            for directory in [self.current_dir, self.recent_dir, self.archived_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            self._logger.error(f"Failed to create directories: {e}")
            raise
    
    def get_current_file_path(self) -> Path:
        """Get path to current log file."""
        return self.current_dir / f"{self.name}.log"
    
    def get_rotated_file_path(self, timestamp: Optional[float] = None) -> Path:
        """Get path for rotated file."""
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        filename = f"{self.name}_{dt.strftime('%Y%m%d_%H%M%S')}.log"
        return self.recent_dir / filename
    
    def get_archived_file_path(self, original_path: Path, compression: CompressionType) -> Path:
        """Get path for archived file."""
        base_name = original_path.stem
        
        if compression == CompressionType.GZIP:
            extension = ".log.gz"
        elif compression == CompressionType.LZ4:
            extension = ".log.lz4"
        elif compression == CompressionType.ZSTD:
            extension = ".log.zst"
        else:
            extension = ".log"
        
        return self.archived_dir / f"{base_name}{extension}"
    
    def atomic_write(self, file_path: Path, content: str, mode: str = 'a') -> bool:
        """Perform atomic write operation."""
        try:
            with self._lock:
                # Write to temporary file first
                temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
                
                try:
                    if mode == 'a' and file_path.exists():
                        # For append mode, copy existing content first
                        shutil.copy2(file_path, temp_path)
                        write_mode = 'a'
                    else:
                        write_mode = 'w'
                    
                    with open(temp_path, write_mode, encoding='utf-8', buffering=8192) as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    
                    # Atomic move
                    if file_path.exists():
                        backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
                        shutil.move(str(file_path), str(backup_path))
                    
                    shutil.move(str(temp_path), str(file_path))
                    
                    # Remove backup
                    if file_path.with_suffix(f"{file_path.suffix}.bak").exists():
                        os.remove(file_path.with_suffix(f"{file_path.suffix}.bak"))
                    
                    return True
                    
                except Exception as e:
                    # Cleanup on error
                    if temp_path.exists():
                        os.remove(temp_path)
                    raise e
                    
        except Exception as e:
            self._logger.error(f"Atomic write failed for {file_path}: {e}")
            return False
    
    def safe_move(self, source: Path, destination: Path) -> bool:
        """Safely move file with error handling."""
        try:
            with self._lock:
                if not source.exists():
                    return False
                
                # Ensure destination directory exists
                destination.parent.mkdir(parents=True, exist_ok=True)
                
                # Atomic move
                shutil.move(str(source), str(destination))
                return True
                
        except Exception as e:
            self._logger.error(f"Safe move failed {source} -> {destination}: {e}")
            return False
    
    def compress_file(self, source_path: Path, compression: CompressionType, level: int = 6) -> Optional[Path]:
        """Compress file using specified algorithm."""
        try:
            destination_path = self.get_archived_file_path(source_path, compression)
            
            if compression == CompressionType.NONE:
                # Just move to archived directory
                return destination_path if self.safe_move(source_path, destination_path) else None
            
            elif compression == CompressionType.GZIP:
                with open(source_path, 'rb') as f_in:
                    with gzip.open(destination_path, 'wb', compresslevel=level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            elif compression == CompressionType.LZ4 and LZ4_AVAILABLE:
                with open(source_path, 'rb') as f_in:
                    with lz4.open(destination_path, 'wb', compression_level=level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            elif compression == CompressionType.ZSTD and ZSTD_AVAILABLE:
                compressor = zstd.ZstdCompressor(level=level)
                with open(source_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        compressor.copy_stream(f_in, f_out)
            
            else:
                self._logger.warning(f"Compression {compression.value} not available, using gzip")
                with open(source_path, 'rb') as f_in:
                    with gzip.open(destination_path.with_suffix('.log.gz'), 'wb', compresslevel=level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                destination_path = destination_path.with_suffix('.log.gz')
            
            # Verify compression succeeded and remove source
            if destination_path.exists() and destination_path.stat().st_size > 0:
                os.remove(source_path)
                return destination_path
            else:
                self._logger.error(f"Compression failed for {source_path}")
                return None
                
        except Exception as e:
            self._logger.error(f"File compression failed: {e}")
            return None
    
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity."""
        try:
            if not file_path.exists():
                return False
            
            # Basic checks
            stat_info = file_path.stat()
            if stat_info.st_size == 0:
                return True  # Empty file is valid
            
            # Try to read file
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    f.read(1024)  # Read first 1KB
            elif file_path.suffix == '.lz4' and LZ4_AVAILABLE:
                with lz4.open(file_path, 'rt') as f:
                    f.read(1024)
            elif file_path.suffix == '.zst' and ZSTD_AVAILABLE:
                decompressor = zstd.ZstdDecompressor()
                with open(file_path, 'rb') as f_in:
                    decompressor.stream_reader(f_in).read(1024)
            else:
                with open(file_path, 'r') as f:
                    f.read(1024)
            
            return True
            
        except Exception as e:
            self._logger.warning(f"File integrity check failed for {file_path}: {e}")
            return False


class TieredFileSink(BaseSink):
    """
    Intelligent tiered file sink with rotation and compression.
    
    Provides sophisticated file management with automatic rotation,
    compression, and multi-tier storage for optimal performance
    and storage efficiency.
    
    Features:
    - Multi-tier storage (current, recent, archived)
    - Configurable rotation strategies
    - Compression with multiple algorithms
    - Automatic cleanup and maintenance
    - File integrity verification
    - Performance optimization
    """
    
    def __init__(self,
                 name: str = "file",
                 log_directory: Union[str, Path] = "./logs",
                 formatter: Optional[Any] = None,
                 rotation_config: Optional[RotationConfig] = None,
                 tier_configs: Optional[Dict[StorageTier, TierConfig]] = None,
                 enable_compression: bool = True,
                 enable_integrity_check: bool = True,
                 maintenance_interval_hours: int = 6,
                 **base_kwargs):
        """
        Initialize tiered file sink.
        
        Args:
            name: Sink name for identification
            log_directory: Base directory for log files
            formatter: Message formatter (defaults to StructuredFormatter)
            rotation_config: File rotation configuration
            tier_configs: Storage tier configurations
            enable_compression: Enable file compression
            enable_integrity_check: Enable file integrity checking
            maintenance_interval_hours: Hours between maintenance runs
            **base_kwargs: Arguments for BaseSink
        """
        # Set up formatter before calling parent __init__
        if formatter is None:
            formatter = StructuredFormatter(
                timestamp_format="iso",
                include_performance_metrics=True,
                include_system_info=True
            )
        
        # Initialize parent
        super().__init__(name=name, formatter=formatter, **base_kwargs)
        
        # Configuration
        self.log_directory = Path(log_directory)
        self.rotation_config = rotation_config or RotationConfig()
        self.enable_compression = enable_compression
        self.enable_integrity_check = enable_integrity_check
        self.maintenance_interval_hours = maintenance_interval_hours
        
        # Default tier configurations
        if tier_configs is None:
            tier_configs = {
                StorageTier.CURRENT: TierConfig(
                    max_files=1,
                    max_age_days=1,
                    compression=CompressionType.NONE
                ),
                StorageTier.RECENT: TierConfig(
                    max_files=10,
                    max_age_days=7,
                    compression=CompressionType.GZIP,
                    compression_level=6
                ),
                StorageTier.ARCHIVED: TierConfig(
                    max_files=100,
                    max_age_days=90,
                    compression=CompressionType.GZIP,
                    compression_level=9
                )
            }
        self.tier_configs = tier_configs
        
        # File management
        self.file_manager = FileManager(self.log_directory, name)
        self.current_file_path = self.file_manager.get_current_file_path()
        
        # State tracking
        self._current_file_size = 0
        self._current_file_lines = 0
        self._last_rotation_time = time.time()
        self._last_maintenance_time = time.time()
        
        # Maintenance scheduling
        self._maintenance_timer: Optional[threading.Timer] = None
        self._maintenance_lock = threading.RLock()
        
        # Performance tracking
        self._write_performance = deque(maxlen=1000)
        self._rotation_times = []
        self._compression_stats = {
            'files_compressed': 0,
            'total_compression_time': 0.0,
            'bytes_before_compression': 0,
            'bytes_after_compression': 0
        }
        
        self._logger.info(f"TieredFileSink '{name}' initialized",
                         extra={
                             'log_directory': str(self.log_directory),
                             'rotation_strategy': self.rotation_config.strategy.value,
                             'compression_enabled': enable_compression,
                             'max_size_mb': self.rotation_config.max_size_mb
                         })
    
    def _initialize_sink(self) -> bool:
        """Initialize tiered file sink."""
        try:
            # Create log directory structure
            self.log_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize current file size if file exists
            if self.current_file_path.exists():
                self._current_file_size = self.current_file_path.stat().st_size
                self._count_file_lines()
            
            # Schedule first maintenance
            self._schedule_maintenance()
            
            # Verify write permissions
            test_file = self.current_file_path.with_suffix('.test')
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                self._logger.error(f"Write permission test failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Tiered file sink initialization failed: {e}")
            return False
    
    def _cleanup_sink(self) -> bool:
        """Cleanup tiered file sink."""
        try:
            # Cancel maintenance timer
            with self._maintenance_lock:
                if self._maintenance_timer:
                    self._maintenance_timer.cancel()
                    self._maintenance_timer = None
            
            # Final maintenance run
            self._perform_maintenance()
            
            self._logger.info(f"TieredFileSink '{self.name}' cleanup completed",
                             extra={
                                 'files_compressed': self._compression_stats['files_compressed'],
                                 'current_file_size': self._current_file_size,
                                 'current_file_lines': self._current_file_lines
                             })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Tiered file sink cleanup failed: {e}")
            return False
    
    def _health_check_sink(self) -> bool:
        """Perform health check on file sink."""
        try:
            # Check if log directory is accessible
            if not self.log_directory.exists():
                return False
            
            # Check write permissions
            if not os.access(self.log_directory, os.W_OK):
                return False
            
            # Check current file status
            if self.current_file_path.exists():
                if not os.access(self.current_file_path, os.W_OK):
                    return False
                
                # Basic integrity check
                if self.enable_integrity_check:
                    return self.file_manager.verify_file_integrity(self.current_file_path)
            
            return True
            
        except Exception as e:
            self._logger.warning(f"File sink health check failed: {e}")
            return False
    
    def _write_message(self, formatted_message: str, message: LogMessage) -> bool:
        """Write formatted message to file."""
        start_time = time.perf_counter()
        
        try:
            # Check if rotation is needed before writing
            if self._should_rotate():
                self._rotate_file()
            
            # Ensure message ends with newline
            if not formatted_message.endswith('\n'):
                formatted_message += '\n'
            
            # Write to current file
            success = self.file_manager.atomic_write(
                self.current_file_path,
                formatted_message,
                mode='a'
            )
            
            if success:
                # Update statistics
                message_bytes = len(formatted_message.encode('utf-8'))
                self._current_file_size += message_bytes
                self._current_file_lines += 1
                
                # Track performance
                elapsed_time = time.perf_counter() - start_time
                self._write_performance.append(elapsed_time)
            
            return success
            
        except Exception as e:
            self._logger.error(f"File write failed: {e}")
            return False
    
    def _should_rotate(self) -> bool:
        """Check if file rotation is needed."""
        try:
            current_time = time.time()
            
            if self.rotation_config.strategy == RotationStrategy.SIZE:
                max_size_bytes = self.rotation_config.max_size_mb * 1024 * 1024
                return self._current_file_size >= max_size_bytes
            
            elif self.rotation_config.strategy == RotationStrategy.TIME:
                interval_seconds = self.rotation_config.rotation_interval_hours * 3600
                return current_time - self._last_rotation_time >= interval_seconds
            
            elif self.rotation_config.strategy == RotationStrategy.DAILY:
                last_rotation_date = datetime.fromtimestamp(self._last_rotation_time).date()
                current_date = datetime.fromtimestamp(current_time).date()
                return current_date > last_rotation_date
            
            elif self.rotation_config.strategy == RotationStrategy.HOURLY:
                last_rotation_hour = datetime.fromtimestamp(self._last_rotation_time).hour
                current_hour = datetime.fromtimestamp(current_time).hour
                last_rotation_date = datetime.fromtimestamp(self._last_rotation_time).date()
                current_date = datetime.fromtimestamp(current_time).date()
                return current_date > last_rotation_date or current_hour > last_rotation_hour
            
            elif self.rotation_config.strategy == RotationStrategy.COUNT:
                return self._current_file_lines >= 10000  # Default count threshold
            
            return False
            
        except Exception as e:
            self._logger.warning(f"Rotation check failed: {e}")
            return False
    
    def _rotate_file(self) -> bool:
        """Rotate current log file."""
        start_time = time.perf_counter()
        
        try:
            if not self.current_file_path.exists() or self._current_file_size == 0:
                return True  # Nothing to rotate
            
            # Generate rotated file path
            rotated_path = self.file_manager.get_rotated_file_path()
            
            # Move current file to recent directory
            success = self.file_manager.safe_move(self.current_file_path, rotated_path)
            
            if success:
                # Reset current file state
                self._current_file_size = 0
                self._current_file_lines = 0
                self._last_rotation_time = time.time()
                
                # Schedule compression if enabled
                if self.enable_compression and self.rotation_config.compression_delay_minutes > 0:
                    compression_timer = threading.Timer(
                        self.rotation_config.compression_delay_minutes * 60,
                        self._compress_file,
                        args=[rotated_path]
                    )
                    compression_timer.daemon = True
                    compression_timer.start()
                elif self.enable_compression:
                    # Compress immediately
                    self._compress_file(rotated_path)
                
                # Track rotation performance
                elapsed_time = time.perf_counter() - start_time
                self._rotation_times.append(elapsed_time)
                if len(self._rotation_times) > 100:
                    self._rotation_times.pop(0)
                
                self._logger.info(f"File rotated: {rotated_path.name}",
                                 extra={
                                     'rotation_time_ms': elapsed_time * 1000,
                                     'rotated_file_size': rotated_path.stat().st_size if rotated_path.exists() else 0
                                 })
            
            return success
            
        except Exception as e:
            self._logger.error(f"File rotation failed: {e}")
            return False
    
    def _compress_file(self, file_path: Path) -> None:
        """Compress a file for archival."""
        try:
            if not file_path.exists():
                return
            
            start_time = time.perf_counter()
            original_size = file_path.stat().st_size
            
            # Get compression config for recent tier
            tier_config = self.tier_configs[StorageTier.RECENT]
            
            # Compress file
            compressed_path = self.file_manager.compress_file(
                file_path,
                tier_config.compression,
                tier_config.compression_level
            )
            
            if compressed_path and compressed_path.exists():
                compressed_size = compressed_path.stat().st_size
                elapsed_time = time.perf_counter() - start_time
                
                # Update compression statistics
                self._compression_stats['files_compressed'] += 1
                self._compression_stats['total_compression_time'] += elapsed_time
                self._compression_stats['bytes_before_compression'] += original_size
                self._compression_stats['bytes_after_compression'] += compressed_size
                
                compression_ratio = compressed_size / original_size if original_size > 0 else 0
                
                self._logger.info(f"File compressed: {compressed_path.name}",
                                 extra={
                                     'compression_type': tier_config.compression.value,
                                     'original_size_mb': original_size / 1024 / 1024,
                                     'compressed_size_mb': compressed_size / 1024 / 1024,
                                     'compression_ratio': compression_ratio,
                                     'compression_time_ms': elapsed_time * 1000
                                 })
            
        except Exception as e:
            self._logger.error(f"File compression failed for {file_path}: {e}")
    
    def _count_file_lines(self) -> None:
        """Count lines in current file."""
        try:
            if self.current_file_path.exists():
                with open(self.current_file_path, 'r', encoding='utf-8') as f:
                    self._current_file_lines = sum(1 for _ in f)
            else:
                self._current_file_lines = 0
        except Exception as e:
            self._logger.warning(f"Line counting failed: {e}")
            self._current_file_lines = 0
    
    def _schedule_maintenance(self) -> None:
        """Schedule maintenance task."""
        try:
            with self._maintenance_lock:
                if self._maintenance_timer:
                    self._maintenance_timer.cancel()
                
                interval_seconds = self.maintenance_interval_hours * 3600
                self._maintenance_timer = threading.Timer(
                    interval_seconds,
                    self._maintenance_callback
                )
                self._maintenance_timer.daemon = True
                self._maintenance_timer.start()
                
        except Exception as e:
            self._logger.warning(f"Maintenance scheduling failed: {e}")
    
    def _maintenance_callback(self) -> None:
        """Maintenance timer callback."""
        try:
            self._perform_maintenance()
            self._schedule_maintenance()  # Schedule next maintenance
            
        except Exception as e:
            self._logger.error(f"Maintenance callback failed: {e}")
    
    def _perform_maintenance(self) -> None:
        """Perform file maintenance tasks."""
        try:
            self._last_maintenance_time = time.time()
            
            # Clean up old files
            self._cleanup_old_files()
            
            # Verify file integrity if enabled
            if self.enable_integrity_check:
                self._verify_files_integrity()
            
            # Compress old files if needed
            self._compress_old_files()
            
            self._logger.debug("File maintenance completed")
            
        except Exception as e:
            self._logger.error(f"File maintenance failed: {e}")
    
    def _cleanup_old_files(self) -> None:
        """Clean up old files based on tier policies."""
        try:
            current_time = time.time()
            
            for tier, config in self.tier_configs.items():
                if tier == StorageTier.CURRENT:
                    continue
                
                # Get tier directory
                if tier == StorageTier.RECENT:
                    tier_dir = self.file_manager.recent_dir
                elif tier == StorageTier.ARCHIVED:
                    tier_dir = self.file_manager.archived_dir
                else:
                    continue
                
                if not tier_dir.exists():
                    continue
                
                # Get all files in tier
                files = list(tier_dir.glob(f"{self.name}_*"))
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Remove files exceeding count limit
                if len(files) > config.max_files:
                    for file_path in files[config.max_files:]:
                        try:
                            os.remove(file_path)
                            self._logger.debug(f"Removed file (count limit): {file_path.name}")
                        except Exception as e:
                            self._logger.warning(f"Failed to remove file {file_path}: {e}")
                
                # Remove files exceeding age limit
                max_age_seconds = config.max_age_days * 24 * 3600
                for file_path in files:
                    try:
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            self._logger.debug(f"Removed file (age limit): {file_path.name}")
                    except Exception as e:
                        self._logger.warning(f"Failed to remove old file {file_path}: {e}")
                        
        except Exception as e:
            self._logger.error(f"File cleanup failed: {e}")
    
    def _verify_files_integrity(self) -> None:
        """Verify integrity of all log files."""
        try:
            corrupted_files = []
            
            for tier_dir in [self.file_manager.current_dir, self.file_manager.recent_dir, self.file_manager.archived_dir]:
                if not tier_dir.exists():
                    continue
                
                for file_path in tier_dir.glob(f"{self.name}*"):
                    if not self.file_manager.verify_file_integrity(file_path):
                        corrupted_files.append(file_path)
            
            if corrupted_files:
                self._logger.warning(f"Found {len(corrupted_files)} corrupted files",
                                   extra={'corrupted_files': [str(f) for f in corrupted_files]})
                
                # Optionally remove corrupted files
                for file_path in corrupted_files:
                    try:
                        backup_path = file_path.with_suffix(f"{file_path.suffix}.corrupted")
                        shutil.move(str(file_path), str(backup_path))
                        self._logger.info(f"Moved corrupted file to {backup_path}")
                    except Exception as e:
                        self._logger.error(f"Failed to handle corrupted file {file_path}: {e}")
                        
        except Exception as e:
            self._logger.error(f"Integrity verification failed: {e}")
    
    def _compress_old_files(self) -> None:
        """Compress old files that haven't been compressed yet."""
        try:
            # Look for uncompressed files in recent directory
            recent_files = list(self.file_manager.recent_dir.glob(f"{self.name}_*.log"))
            
            for file_path in recent_files:
                try:
                    # Check if file is old enough to compress
                    file_age_minutes = (time.time() - file_path.stat().st_mtime) / 60
                    if file_age_minutes >= self.rotation_config.compression_delay_minutes:
                        self._compress_file(file_path)
                        
                except Exception as e:
                    self._logger.warning(f"Failed to compress old file {file_path}: {e}")
                    
        except Exception as e:
            self._logger.error(f"Old file compression failed: {e}")
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get comprehensive file sink statistics."""
        base_stats = self.get_stats().to_dict()
        
        # File-specific statistics
        file_stats = {
            'current_file_size_mb': self._current_file_size / 1024 / 1024,
            'current_file_lines': self._current_file_lines,
            'last_rotation_time': self._last_rotation_time,
            'last_maintenance_time': self._last_maintenance_time,
            'log_directory': str(self.log_directory),
            'rotation_strategy': self.rotation_config.strategy.value,
            'compression_enabled': self.enable_compression
        }
        
        # Performance statistics
        if self._write_performance:
            file_stats['write_performance'] = {
                'average_write_time_us': (sum(self._write_performance) / len(self._write_performance)) * 1_000_000,
                'max_write_time_us': max(self._write_performance) * 1_000_000,
                'samples': len(self._write_performance)
            }
        
        if self._rotation_times:
            file_stats['rotation_performance'] = {
                'average_rotation_time_ms': (sum(self._rotation_times) / len(self._rotation_times)) * 1000,
                'max_rotation_time_ms': max(self._rotation_times) * 1000,
                'rotation_count': len(self._rotation_times)
            }
        
        # Compression statistics
        file_stats['compression_stats'] = self._compression_stats.copy()
        if self._compression_stats['bytes_before_compression'] > 0:
            file_stats['compression_stats']['overall_compression_ratio'] = (
                self._compression_stats['bytes_after_compression'] / 
                self._compression_stats['bytes_before_compression']
            )
        
        # File count statistics
        try:
            file_stats['file_counts'] = {
                'current': len(list(self.file_manager.current_dir.glob(f"{self.name}*"))),
                'recent': len(list(self.file_manager.recent_dir.glob(f"{self.name}*"))),
                'archived': len(list(self.file_manager.archived_dir.glob(f"{self.name}*")))
            }
        except Exception:
            file_stats['file_counts'] = {'error': 'failed_to_count'}
        
        # Merge with base stats
        base_stats.update(file_stats)
        return base_stats
    
    def force_rotation(self) -> bool:
        """Force immediate file rotation."""
        return self._rotate_file()
    
    def force_maintenance(self) -> None:
        """Force immediate maintenance run."""
        self._perform_maintenance()


# Module exports
__all__ = [
    'TieredFileSink',
    'RotationStrategy',
    'CompressionType',
    'StorageTier',
    'TierConfig',
    'RotationConfig'
]