#!/usr/bin/env python3
"""
Platform-Aware Logging Configuration for Arena Bot
Provides cross-platform Unicode-safe logging with proper encoding handling.
"""

import logging
import sys
import os
import locale
from typing import Optional, TextIO
from pathlib import Path


class PlatformAwareFormatter(logging.Formatter):
    """
    Custom formatter that handles Unicode characters safely across platforms.
    Falls back to ASCII alternatives when Unicode isn't supported.
    """
    
    # Unicode to ASCII fallback mappings for problematic characters
    UNICODE_FALLBACKS = {
        '✅': '[OK]',
        '❌': '[ERROR]', 
        '⚠️': '[WARN]',
        '🎯': '[TARGET]',
        '🔄': '[LOADING]',
        '⏳': '[WAIT]',
        '🚨': '[CRITICAL]',
        '💓': '[HEARTBEAT]',
        '💔': '[FAILED]',
        '📁': '[FOLDER]',
        '📂': '[DIR]',
        '📖': '[READ]',
        '🎮': '[GAME]',
        '🏠': '[HOME]',
        '⚔️': '[BATTLE]',
        '🛡️': '[SHIELD]',
        '👑': '[HERO]',
        '🎨': '[UI]',
        '🌐': '[NET]',
        '📱': '[MOBILE]',
        '🏗️': '[ARCH]',
        '🧩': '[MODULE]',
        '✨': '[PREMIUM]',
        '🔐': '[LOGIN]',
        '📚': '[COLLECTION]',
        '🏆': '[TOURNAMENT]',
        '🥊': '[BATTLEGROUNDS]',
        '🗺️': '[ADVENTURE]',
        '🍺': '[TAVERN]',
        '🛒': '[SHOP]',
        '❓': '[UNKNOWN]',
        '█': '#',
        '🎯': '*',  # Duplicate key - using simpler fallback
    }
    
    def __init__(self, fmt=None, datefmt=None, console_encoding='utf-8'):
        """
        Initialize the formatter with encoding-aware settings.
        
        Args:
            fmt: Log format string
            datefmt: Date format string  
            console_encoding: Target console encoding
        """
        super().__init__(fmt, datefmt)
        self.console_encoding = console_encoding
        self.supports_unicode = self._test_unicode_support()
    
    def _test_unicode_support(self) -> bool:
        """
        Test if the current console supports Unicode output.
        
        Returns:
            bool: True if Unicode is supported, False otherwise
        """
        try:
            # Try encoding a test Unicode character
            test_char = '✅'
            test_char.encode(self.console_encoding)
            return True
        except (UnicodeEncodeError, LookupError):
            return False
    
    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize log message by replacing Unicode characters with ASCII fallbacks.
        
        Args:
            message: Original log message
            
        Returns:
            str: Sanitized message safe for the target encoding
        """
        if self.supports_unicode:
            return message
        
        # Replace Unicode characters with ASCII fallbacks
        sanitized = message
        for unicode_char, ascii_fallback in self.UNICODE_FALLBACKS.items():
            sanitized = sanitized.replace(unicode_char, ascii_fallback)
        
        # Handle any remaining non-ASCII characters
        try:
            sanitized.encode(self.console_encoding)
            return sanitized
        except UnicodeEncodeError:
            # Final fallback: encode with error replacement
            return sanitized.encode(self.console_encoding, errors='replace').decode(self.console_encoding)
    
    def format(self, record):
        """Format the log record with safe Unicode handling."""
        # Get the base formatted message
        formatted_message = super().format(record)
        
        # Sanitize for console compatibility
        return self._sanitize_message(formatted_message)


class SafeStreamHandler(logging.StreamHandler):
    """
    Stream handler with safe encoding and error handling.
    Automatically detects and adapts to console encoding capabilities.
    """
    
    def __init__(self, stream: Optional[TextIO] = None):
        """
        Initialize handler with encoding detection.
        
        Args:
            stream: Output stream (defaults to sys.stderr)
        """
        super().__init__(stream)
        self.console_encoding = self._detect_console_encoding()
        self.supports_unicode = self._test_console_unicode()
        
        # Set up the formatter with detected encoding
        formatter = PlatformAwareFormatter(
            fmt='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
            datefmt='%H:%M:%S',
            console_encoding=self.console_encoding
        )
        self.setFormatter(formatter)
    
    def _detect_console_encoding(self) -> str:
        """
        Detect the best encoding to use for console output.
        
        Returns:
            str: Encoding name to use
        """
        # Try various methods to detect console encoding
        encodings_to_try = []
        
        # Method 1: Check stdout encoding
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
            encodings_to_try.append(sys.stdout.encoding)
        
        # Method 2: Check stderr encoding  
        if hasattr(sys.stderr, 'encoding') and sys.stderr.encoding:
            encodings_to_try.append(sys.stderr.encoding)
        
        # Method 3: System locale
        try:
            locale_encoding = locale.getpreferredencoding()
            if locale_encoding:
                encodings_to_try.append(locale_encoding)
        except Exception:
            pass
        
        # Method 4: Platform defaults
        if sys.platform.startswith('win'):
            encodings_to_try.extend(['cp1252', 'cp437', 'utf-8'])
        else:
            encodings_to_try.extend(['utf-8', 'latin1'])
        
        # Test each encoding
        for encoding in encodings_to_try:
            try:
                # Test if we can encode a problematic character
                '✅'.encode(encoding)
                return encoding
            except (UnicodeEncodeError, LookupError):
                continue
        
        # Final fallback
        return 'ascii'
    
    def _test_console_unicode(self) -> bool:
        """Test if console supports Unicode characters."""
        try:
            test_msg = '✅ Unicode Test'
            test_msg.encode(self.console_encoding)
            return True
        except UnicodeEncodeError:
            return False
    
    def emit(self, record):
        """Emit a log record with safe encoding handling."""
        try:
            super().emit(record)
        except UnicodeEncodeError as e:
            # Fallback: create a safe version of the record
            safe_record = logging.makeLogRecord(record.__dict__)
            safe_record.msg = str(record.msg).encode(self.console_encoding, errors='replace').decode(self.console_encoding)
            
            try:
                super().emit(safe_record)
            except Exception:
                # Ultimate fallback: write to stderr with basic info
                sys.stderr.write(f"[LOGGING ERROR] Could not emit log: {e}\n")


class SafeFileHandler(logging.FileHandler):
    """
    File handler with UTF-8 encoding and safe error handling.
    Always uses UTF-8 for file output regardless of console encoding.
    """
    
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False, errors='replace'):
        """
        Initialize file handler with UTF-8 encoding.
        
        Args:
            filename: Log file path
            mode: File open mode
            encoding: File encoding (always UTF-8)
            delay: Whether to delay file opening
            errors: How to handle encoding errors
        """
        # Ensure parent directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Always use UTF-8 for file output with error replacement
        super().__init__(filename, mode, encoding='utf-8', delay=delay, errors='replace')
        
        # Set up formatter for file output (can use full Unicode)
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up platform-aware logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        console_output: Whether to enable console output
        
    Returns:
        logging.Logger: Configured root logger
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(level)
    
    # Add console handler if requested
    if console_output:
        console_handler = SafeStreamHandler()
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            file_handler = SafeFileHandler(log_file)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"[WARN] Could not create log file {log_file}: {e}")
    
    return root_logger


def get_platform_info() -> dict:
    """
    Get detailed platform information for debugging.
    
    Returns:
        dict: Platform information
    """
    info = {
        'platform': sys.platform,
        'python_version': sys.version,
        'encoding_default': sys.getdefaultencoding(),
        'filesystem_encoding': sys.getfilesystemencoding(),
    }
    
    # Console encoding info
    if hasattr(sys.stdout, 'encoding'):
        info['stdout_encoding'] = sys.stdout.encoding
    if hasattr(sys.stderr, 'encoding'):
        info['stderr_encoding'] = sys.stderr.encoding
    
    # Locale info
    try:
        info['locale_encoding'] = locale.getpreferredencoding()
        info['locale_info'] = locale.getlocale()
    except Exception as e:
        info['locale_error'] = str(e)
    
    return info


def test_unicode_support():
    """Test Unicode support and print diagnostics."""
    print("=== UNICODE SUPPORT TEST ===")
    
    # Platform info
    platform_info = get_platform_info()
    for key, value in platform_info.items():
        print(f"{key}: {value}")
    
    print("\n=== TESTING UNICODE CHARACTERS ===")
    
    test_chars = ['✅', '❌', '⚠️', '🎯', '💓', '█']
    
    for char in test_chars:
        try:
            print(f"Testing '{char}': ", end='')
            char.encode('utf-8')
            print("UTF-8 OK", end='')
            
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
                char.encode(sys.stdout.encoding)
                print(f", {sys.stdout.encoding} OK")
            else:
                print(", Console encoding unknown")
                
        except UnicodeEncodeError as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    """Run diagnostic tests."""
    test_unicode_support()
    
    print("\n=== TESTING LOGGING SYSTEM ===")
    
    # Set up logging
    logger = setup_logging(level=logging.INFO, console_output=True)
    
    # Test various log messages with Unicode
    logger.info("✅ Testing Unicode logging")
    logger.warning("⚠️ Warning with emoji")
    logger.error("❌ Error with emoji")
    logger.info("🎯 Game state: Arena Draft")
    logger.info("💓 Heartbeat OK")
    
    print("\n=== LOGGING TEST COMPLETE ===")