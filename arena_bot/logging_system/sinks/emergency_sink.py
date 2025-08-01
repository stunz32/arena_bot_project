"""
Emergency Sink for S-Tier Logging System.

This module provides ultra-reliable emergency logging when all other sinks fail.
Designed for maximum reliability with minimal dependencies and failure modes.

Features:
- Multiple fallback destinations (stderr, syslog, files, memory)
- Minimal dependencies and resource usage
- Atomic operations and crash-safe writes
- Emergency protocol activation
- System health monitoring integration
- Last-resort message preservation
- Performance optimized for emergency scenarios
"""

import os
import sys
import time
import logging
import threading
import tempfile
import traceback
import syslog
import json
from typing import Any, Dict, List, Optional, Union, TextIO
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime

# Import from our components
from .base_sink import BaseSink, SinkState, ErrorHandlingStrategy
from ..core.hybrid_async_queue import LogMessage


class EmergencyDestination(Enum):
    """Emergency logging destinations."""
    STDERR = "stderr"           # Standard error output
    STDOUT = "stdout"           # Standard output (fallback)
    SYSLOG = "syslog"          # System log
    TEMP_FILE = "temp_file"    # Temporary file
    MEMORY = "memory"          # In-memory buffer
    KERNEL_LOG = "kernel_log"  # Kernel log (/dev/kmsg on Linux)


class EmergencyTrigger(Enum):
    """Conditions that trigger emergency mode."""
    SINK_FAILURE = "sink_failure"           # Other sinks failing
    SYSTEM_OVERLOAD = "system_overload"     # System resource exhaustion
    DISK_FULL = "disk_full"                # Disk space exhausted
    NETWORK_FAILURE = "network_failure"     # Network connectivity issues
    MEMORY_PRESSURE = "memory_pressure"     # Memory pressure
    MANUAL_ACTIVATION = "manual_activation" # Manual emergency activation
    CRITICAL_ERROR = "critical_error"       # Critical system errors


@dataclass
class EmergencyConfig:
    """Emergency sink configuration."""
    destinations: List[EmergencyDestination] = field(default_factory=lambda: [
        EmergencyDestination.STDERR,
        EmergencyDestination.SYSLOG,
        EmergencyDestination.TEMP_FILE,
        EmergencyDestination.MEMORY
    ])
    memory_buffer_size: int = 10000         # Max messages in memory buffer
    temp_file_max_size_mb: float = 10.0     # Max size of temp files
    temp_file_rotation_count: int = 3       # Number of temp files to rotate
    enable_kernel_log: bool = False         # Enable kernel log (requires root)
    max_message_length: int = 8192          # Maximum message length
    enable_compression: bool = False        # Disable compression for reliability
    flush_immediately: bool = True          # Flush after each message
    include_stacktrace: bool = True         # Include stack traces
    fallback_format: str = "simple"        # Simple/JSON/minimal


class MemoryBuffer:
    """Thread-safe circular memory buffer for emergency logging."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.messages = deque(maxlen=max_size)
        self.dropped_count = 0
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.MemoryBuffer")
    
    def add_message(self, message: str, timestamp: float) -> None:
        """Add message to buffer."""
        try:
            with self._lock:
                if len(self.messages) >= self.max_size:
                    self.dropped_count += 1
                
                self.messages.append({
                    'message': message,
                    'timestamp': timestamp,
                    'sequence': len(self.messages) + self.dropped_count
                })
                
        except Exception as e:
            # Even this can't fail
            try:
                sys.stderr.write(f"EMERGENCY: Memory buffer add failed: {e}\n")
                sys.stderr.flush()
            except Exception:
                pass
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from buffer."""
        try:
            with self._lock:
                if limit is None:
                    return list(self.messages)
                else:
                    return list(self.messages)[-limit:]
        except Exception:
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        try:
            with self._lock:
                return {
                    'current_size': len(self.messages),
                    'max_size': self.max_size,
                    'dropped_count': self.dropped_count,
                    'total_messages': len(self.messages) + self.dropped_count
                }
        except Exception:
            return {'error': 'stats_failed'}
    
    def clear(self) -> None:
        """Clear buffer."""
        try:
            with self._lock:
                self.messages.clear()
        except Exception:
            pass


class EmergencySink(BaseSink):
    """
    Ultra-reliable emergency logging sink.
    
    Provides last-resort logging when all other sinks fail. Designed
    for maximum reliability with minimal dependencies and multiple
    fallback destinations.
    
    Features:
    - Multiple fallback destinations
    - Crash-safe atomic operations
    - Minimal resource usage
    - System integration (syslog, kernel log)
    - Memory buffer for critical messages
    - Emergency protocol activation
    """
    
    def __init__(self,
                 name: str = "emergency",
                 config: Optional[EmergencyConfig] = None,
                 formatter: Optional[Any] = None,
                 emergency_triggers: Optional[List[EmergencyTrigger]] = None,
                 auto_activate_on_errors: int = 5,
                 **base_kwargs):
        """
        Initialize emergency sink.
        
        Args:
            name: Sink name for identification
            config: Emergency sink configuration
            formatter: Message formatter (minimal for reliability)
            emergency_triggers: Conditions that activate emergency mode
            auto_activate_on_errors: Auto-activate after N consecutive errors
            **base_kwargs: Arguments for BaseSink
        """
        # Use minimal formatter for maximum reliability
        if formatter is None:
            formatter = self._create_minimal_formatter()
        
        # Initialize parent with special error handling for emergency sink
        base_kwargs.setdefault('error_strategy', ErrorHandlingStrategy.LOG_AND_CONTINUE)
        base_kwargs.setdefault('max_retries', 1)  # Minimal retries for speed
        super().__init__(name=name, formatter=formatter, **base_kwargs)
        
        # Configuration
        self.config = config or EmergencyConfig()
        self.emergency_triggers = emergency_triggers or [
            EmergencyTrigger.SINK_FAILURE,
            EmergencyTrigger.SYSTEM_OVERLOAD,
            EmergencyTrigger.CRITICAL_ERROR
        ]
        self.auto_activate_on_errors = auto_activate_on_errors
        
        # Emergency state
        self.emergency_active = False
        self.activation_time: Optional[float] = None
        self.activation_reason: Optional[str] = None
        self.consecutive_errors = 0
        
        # Memory buffer
        self.memory_buffer = MemoryBuffer(self.config.memory_buffer_size)
        
        # Destination handlers
        self._destination_handlers = {
            EmergencyDestination.STDERR: self._write_to_stderr,
            EmergencyDestination.STDOUT: self._write_to_stdout,
            EmergencyDestination.SYSLOG: self._write_to_syslog,
            EmergencyDestination.TEMP_FILE: self._write_to_temp_file,
            EmergencyDestination.MEMORY: self._write_to_memory,
            EmergencyDestination.KERNEL_LOG: self._write_to_kernel_log
        }
        
        # Temp file management
        self.temp_file_path: Optional[Path] = None
        self.temp_file_handle: Optional[TextIO] = None
        self.temp_file_size = 0
        self.temp_file_rotation_index = 0
        self.temp_file_lock = threading.RLock()
        
        # Syslog setup
        self.syslog_initialized = False
        self._setup_syslog()
        
        # Statistics
        self._emergency_activations = 0
        self._messages_written = 0
        self._write_failures = 0
        self._destination_stats: Dict[str, int] = {}
        
        # Initialize destination stats
        for dest in self.config.destinations:
            self._destination_stats[dest.value] = 0
        
        self._logger.info(f"EmergencySink '{name}' initialized",
                         extra={
                             'destinations': [d.value for d in self.config.destinations],
                             'memory_buffer_size': self.config.memory_buffer_size,
                             'auto_activate_threshold': auto_activate_on_errors,
                             'emergency_triggers': [t.value for t in self.emergency_triggers]
                         })
    
    def _create_minimal_formatter(self) -> Any:
        """Create minimal formatter for emergency use."""
        class MinimalFormatter:
            def format(self, message: LogMessage) -> str:
                try:
                    # Create minimal format for maximum reliability
                    timestamp = datetime.fromtimestamp(message.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    level_name = {
                        10: "DEBUG", 20: "INFO", 30: "WARNING", 
                        40: "ERROR", 50: "CRITICAL"
                    }.get(message.level, f"L{message.level}")
                    
                    # Basic format
                    basic_msg = f"{timestamp} {level_name} [{message.logger_name}] {message.message}"
                    
                    # Add correlation ID if available
                    if message.correlation_id:
                        basic_msg += f" [ID:{message.correlation_id}]"
                    
                    # Add error info if available
                    if message.error and isinstance(message.error, dict):
                        error_type = message.error.get('exception_type', 'Unknown')
                        error_msg = message.error.get('message', 'No details')
                        basic_msg += f" [ERROR:{error_type}:{error_msg}]"
                    
                    return basic_msg
                    
                except Exception:
                    # Ultimate fallback
                    return f"{time.time()} EMERGENCY [{message.logger_name}] {message.message}"
        
        return MinimalFormatter()
    
    def _initialize_sink(self) -> bool:
        """Initialize emergency sink."""
        try:
            # Test all destinations
            self._test_destinations()
            
            # Create temp file if needed
            if EmergencyDestination.TEMP_FILE in self.config.destinations:
                self._initialize_temp_file()
            
            return True
            
        except Exception as e:
            # Even initialization failure shouldn't stop emergency sink
            try:
                sys.stderr.write(f"EMERGENCY: Sink initialization warning: {e}\n")
                sys.stderr.flush()
            except Exception:
                pass
            return True  # Always succeed for emergency sink
    
    def _cleanup_sink(self) -> bool:
        """Cleanup emergency sink."""
        try:
            # Close temp file
            with self.temp_file_lock:
                if self.temp_file_handle:
                    try:
                        self.temp_file_handle.close()
                    except Exception:
                        pass
                    self.temp_file_handle = None
            
            # Close syslog
            if self.syslog_initialized:
                try:
                    syslog.closelog()
                except Exception:
                    pass
            
            # Dump memory buffer to stderr as final act
            self._dump_memory_buffer()
            
            return True
            
        except Exception:
            return True  # Always succeed for emergency sink
    
    def _health_check_sink(self) -> bool:
        """Emergency sink is always healthy."""
        return True  # Emergency sink never reports unhealthy
    
    def _write_message(self, formatted_message: str, message: LogMessage) -> bool:
        """Write message to emergency destinations."""
        # Check for auto-activation
        if not self.emergency_active:
            self._check_auto_activation()
        
        # Limit message length for reliability
        if len(formatted_message) > self.config.max_message_length:
            formatted_message = formatted_message[:self.config.max_message_length] + "... [TRUNCATED]"
        
        success_count = 0
        
        # Try each destination
        for destination in self.config.destinations:
            try:
                handler = self._destination_handlers.get(destination)
                if handler and handler(formatted_message, message):
                    success_count += 1
                    self._destination_stats[destination.value] += 1
                    
            except Exception as e:
                # Even emergency logging can fail, but we continue
                try:
                    sys.stderr.write(f"EMERGENCY: Destination {destination.value} failed: {e}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
        
        # Update statistics
        if success_count > 0:
            self._messages_written += 1
            self.consecutive_errors = 0
        else:
            self._write_failures += 1
            self.consecutive_errors += 1
        
        # Always return True for emergency sink (it never "fails")
        return True
    
    def _check_auto_activation(self) -> None:
        """Check if emergency mode should be auto-activated."""
        if self.consecutive_errors >= self.auto_activate_on_errors:
            self.activate_emergency_mode(EmergencyTrigger.SINK_FAILURE, 
                                       f"Auto-activated after {self.consecutive_errors} consecutive errors")
    
    def _test_destinations(self) -> None:
        """Test all emergency destinations."""
        test_message = f"EMERGENCY SINK TEST - {datetime.now().isoformat()}"
        
        for destination in self.config.destinations:
            try:
                handler = self._destination_handlers.get(destination)
                if handler:
                    # Create minimal test message
                    test_log_message = LogMessage(
                        timestamp=time.time(),
                        level=logging.INFO,
                        logger_name="emergency_test",
                        message=test_message,
                        correlation_id="test",
                        thread_id="test",
                        process_id=os.getpid()
                    )
                    handler(test_message, test_log_message)
                    
            except Exception as e:
                # Log test failure but don't fail initialization
                try:
                    sys.stderr.write(f"EMERGENCY: Destination test failed for {destination.value}: {e}\n")
                    sys.stderr.flush()
                except Exception:
                    pass
    
    def _write_to_stderr(self, message: str, log_message: LogMessage) -> bool:
        """Write to standard error."""
        try:
            sys.stderr.write(f"EMERGENCY: {message}\n")
            if self.config.flush_immediately:
                sys.stderr.flush()
            return True
        except Exception:
            return False
    
    def _write_to_stdout(self, message: str, log_message: LogMessage) -> bool:
        """Write to standard output."""
        try:
            sys.stdout.write(f"EMERGENCY: {message}\n")
            if self.config.flush_immediately:
                sys.stdout.flush()
            return True
        except Exception:
            return False
    
    def _write_to_syslog(self, message: str, log_message: LogMessage) -> bool:
        """Write to system log."""
        try:
            if not self.syslog_initialized:
                return False
            
            # Map log levels to syslog levels
            level_map = {
                logging.DEBUG: syslog.LOG_DEBUG,
                logging.INFO: syslog.LOG_INFO,
                logging.WARNING: syslog.LOG_WARNING,
                logging.ERROR: syslog.LOG_ERR,
                logging.CRITICAL: syslog.LOG_CRIT
            }
            
            syslog_level = level_map.get(log_message.level, syslog.LOG_INFO)
            syslog.syslog(syslog_level, f"EMERGENCY: {message}")
            return True
            
        except Exception:
            return False
    
    def _write_to_temp_file(self, message: str, log_message: LogMessage) -> bool:
        """Write to temporary file."""
        try:
            with self.temp_file_lock:
                # Initialize temp file if needed
                if not self.temp_file_handle:
                    self._initialize_temp_file()
                
                if not self.temp_file_handle:
                    return False
                
                # Write message
                line = f"{message}\n"
                self.temp_file_handle.write(line)
                
                if self.config.flush_immediately:
                    self.temp_file_handle.flush()
                    os.fsync(self.temp_file_handle.fileno())
                
                self.temp_file_size += len(line.encode('utf-8'))
                
                # Check for rotation
                max_size_bytes = self.config.temp_file_max_size_mb * 1024 * 1024
                if self.temp_file_size >= max_size_bytes:
                    self._rotate_temp_file()
                
                return True
                
        except Exception:
            return False
    
    def _write_to_memory(self, message: str, log_message: LogMessage) -> bool:
        """Write to memory buffer."""
        try:
            self.memory_buffer.add_message(message, log_message.timestamp)
            return True
        except Exception:
            return False
    
    def _write_to_kernel_log(self, message: str, log_message: LogMessage) -> bool:
        """Write to kernel log (/dev/kmsg on Linux)."""
        try:
            if not self.config.enable_kernel_log:
                return False
            
            # Only available on Linux and requires privileges
            if not os.path.exists('/dev/kmsg'):
                return False
            
            # Map to kernel log levels
            level_map = {
                logging.DEBUG: 7,    # KERN_DEBUG
                logging.INFO: 6,     # KERN_INFO  
                logging.WARNING: 4,  # KERN_WARNING
                logging.ERROR: 3,    # KERN_ERR
                logging.CRITICAL: 2  # KERN_CRIT
            }
            
            kern_level = level_map.get(log_message.level, 6)
            
            with open('/dev/kmsg', 'w') as kmsg:
                kmsg.write(f"<{kern_level}>EMERGENCY: {message}\n")
            
            return True
            
        except Exception:
            return False
    
    def _setup_syslog(self) -> None:
        """Initialize syslog connection."""
        try:
            syslog.openlog(f"s-tier-logging-{self.name}", syslog.LOG_PID, syslog.LOG_USER)
            self.syslog_initialized = True
        except Exception:
            self.syslog_initialized = False
    
    def _initialize_temp_file(self) -> None:
        """Initialize temporary file for logging."""
        try:
            if self.temp_file_handle:
                return  # Already initialized
            
            # Create temp file in system temp directory
            temp_dir = Path(tempfile.gettempdir()) / "s-tier-logging-emergency"
            temp_dir.mkdir(exist_ok=True)
            
            # Generate temp file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emergency_{self.name}_{timestamp}_{self.temp_file_rotation_index}.log"
            self.temp_file_path = temp_dir / filename
            
            # Open file
            self.temp_file_handle = open(self.temp_file_path, 'w', encoding='utf-8', buffering=1)
            self.temp_file_size = 0
            
            # Write header
            header = f"# S-Tier Emergency Log - Started {datetime.now().isoformat()}\n"
            self.temp_file_handle.write(header)
            self.temp_file_size += len(header.encode('utf-8'))
            
        except Exception as e:
            self.temp_file_handle = None
            try:
                sys.stderr.write(f"EMERGENCY: Temp file initialization failed: {e}\n")
                sys.stderr.flush()
            except Exception:
                pass
    
    def _rotate_temp_file(self) -> None:
        """Rotate temporary file."""
        try:
            # Close current file
            if self.temp_file_handle:
                self.temp_file_handle.close()
                self.temp_file_handle = None
            
            # Increment rotation index
            self.temp_file_rotation_index += 1
            
            # Clean old files if needed
            if self.temp_file_rotation_index > self.config.temp_file_rotation_count:
                self._cleanup_old_temp_files()
            
            # Initialize new file
            self._initialize_temp_file()
            
        except Exception as e:
            try:
                sys.stderr.write(f"EMERGENCY: Temp file rotation failed: {e}\n")
                sys.stderr.flush()
            except Exception:
                pass
    
    def _cleanup_old_temp_files(self) -> None:
        """Clean up old temporary files."""
        try:
            if not self.temp_file_path:
                return
            
            temp_dir = self.temp_file_path.parent
            pattern = f"emergency_{self.name}_*"
            
            # Get all matching files
            temp_files = list(temp_dir.glob(pattern))
            temp_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove oldest files, keep only the configured count
            if len(temp_files) > self.config.temp_file_rotation_count:
                for old_file in temp_files[:-self.config.temp_file_rotation_count]:
                    try:
                        old_file.unlink()
                    except Exception:
                        pass
                        
        except Exception:
            pass
    
    def _dump_memory_buffer(self) -> None:
        """Dump memory buffer to stderr during cleanup."""
        try:
            messages = self.memory_buffer.get_messages()
            if messages:
                sys.stderr.write(f"\n=== EMERGENCY MEMORY BUFFER DUMP ({len(messages)} messages) ===\n")
                for msg_data in messages[-100:]:  # Last 100 messages
                    sys.stderr.write(f"{msg_data['message']}\n")
                sys.stderr.write("=== END EMERGENCY MEMORY BUFFER DUMP ===\n")
                sys.stderr.flush()
        except Exception:
            pass
    
    def activate_emergency_mode(self, trigger: EmergencyTrigger, reason: str) -> None:
        """Manually activate emergency mode."""
        try:
            if not self.emergency_active:
                self.emergency_active = True
                self.activation_time = time.time()
                self.activation_reason = f"{trigger.value}: {reason}"
                self._emergency_activations += 1
                
                # Log activation
                activation_msg = f"EMERGENCY MODE ACTIVATED - Trigger: {trigger.value}, Reason: {reason}"
                
                # Write to all destinations immediately
                for destination in self.config.destinations:
                    try:
                        handler = self._destination_handlers.get(destination)
                        if handler:
                            # Create activation log message
                            log_message = LogMessage(
                                timestamp=time.time(),
                                level=logging.CRITICAL,
                                logger_name="emergency_system",
                                message=activation_msg,
                                correlation_id="emergency_activation",
                                thread_id="emergency",
                                process_id=os.getpid()
                            )
                            handler(activation_msg, log_message)
                    except Exception:
                        pass
                
                self._logger.critical(activation_msg)
                
        except Exception as e:
            # Even activation failure is logged
            try:
                sys.stderr.write(f"EMERGENCY: Activation failed: {e}\n")
                sys.stderr.flush()
            except Exception:
                pass
    
    def deactivate_emergency_mode(self) -> None:
        """Deactivate emergency mode."""
        try:
            if self.emergency_active:
                duration = time.time() - (self.activation_time or 0)
                deactivation_msg = f"EMERGENCY MODE DEACTIVATED - Duration: {duration:.1f}s, Messages: {self._messages_written}"
                
                self.emergency_active = False
                self.activation_time = None
                self.activation_reason = None
                
                self._logger.info(deactivation_msg)
                
        except Exception:
            pass
    
    def get_emergency_stats(self) -> Dict[str, Any]:
        """Get comprehensive emergency sink statistics."""
        base_stats = self.get_stats().to_dict()
        
        # Emergency-specific statistics
        emergency_stats = {
            'emergency_active': self.emergency_active,
            'activation_time': self.activation_time,
            'activation_reason': self.activation_reason,
            'emergency_activations': self._emergency_activations,
            'messages_written': self._messages_written,
            'write_failures': self._write_failures,
            'consecutive_errors': self.consecutive_errors,
            'auto_activate_threshold': self.auto_activate_on_errors,
            'destination_stats': self._destination_stats.copy(),
            'memory_buffer_stats': self.memory_buffer.get_stats(),
            'temp_file_info': {
                'path': str(self.temp_file_path) if self.temp_file_path else None,
                'size_mb': self.temp_file_size / 1024 / 1024,
                'rotation_index': self.temp_file_rotation_index
            },
            'syslog_initialized': self.syslog_initialized
        }
        
        # Configuration summary
        emergency_stats['configuration'] = {
            'destinations': [d.value for d in self.config.destinations],
            'memory_buffer_size': self.config.memory_buffer_size,
            'temp_file_max_size_mb': self.config.temp_file_max_size_mb,
            'flush_immediately': self.config.flush_immediately,
            'max_message_length': self.config.max_message_length
        }
        
        # Merge with base stats
        base_stats.update(emergency_stats)
        return base_stats
    
    def get_memory_buffer_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from memory buffer."""
        return self.memory_buffer.get_messages(limit)
    
    def clear_memory_buffer(self) -> None:
        """Clear memory buffer."""
        self.memory_buffer.clear()
        self._logger.info("Emergency memory buffer cleared")
    
    def get_temp_file_path(self) -> Optional[Path]:
        """Get current temporary file path."""
        return self.temp_file_path
    
    def force_temp_file_rotation(self) -> bool:
        """Force rotation of temporary file."""
        try:
            with self.temp_file_lock:
                self._rotate_temp_file()
            return True
        except Exception as e:
            self._logger.error(f"Force temp file rotation failed: {e}")
            return False


# Module exports
__all__ = [
    'EmergencySink',
    'EmergencyConfig',
    'EmergencyDestination',
    'EmergencyTrigger',
    'MemoryBuffer'
]