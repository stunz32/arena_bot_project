"""
Simple logging configuration for Arena Bot.

Following CLAUDE.md principles - minimal, focused, and easy to understand.
"""

import logging
import logging.handlers
import io
import sys
from pathlib import Path
from datetime import datetime


class SafeUnicodeFormatter(logging.Formatter):
    """
    Enhanced formatter that safely handles Unicode characters.
    Converts problematic Unicode to ASCII-safe alternatives for maximum compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Unicode to ASCII-safe mappings for common characters
        self.unicode_replacements = {
            '✅': '[OK]',
            '❌': '[ERROR]', 
            '⚠️': '[WARNING]',
            '🎯': '[TARGET]',
            '📁': '[FOLDER]',
            '🔍': '[SEARCH]',
            '💔': '[FAILED]',
            '💓': '[HEARTBEAT]',
            '🚀': '[START]',
            '⏸️': '[STOP]',
            '🎮': '[GAME]',
            '📂': '[DIR]',
            '🔄': '[PROCESSING]',
            '📖': '[READING]',
            '🏠': '[MENU]',
            '⚔️': '[BATTLE]',
            '📚': '[COLLECTION]',
            '🏆': '[TOURNAMENT]',
            '🥊': '[BATTLEGROUNDS]',
            '🗺️': '[ADVENTURE]',
            '🍺': '[TAVERN_BRAWL]',
            '🛒': '[SHOP]',
            '👑': '[HERO]',
            '📋': '[DECK]',
            '✨': '[PREMIUM]',
            '🔐': '[LOGIN]',
        }
    
    def format(self, record):
        """Format log record with safe Unicode handling."""
        try:
            # Get the formatted message
            formatted = super().format(record)
            
            # Replace problematic Unicode characters
            for unicode_char, ascii_replacement in self.unicode_replacements.items():
                formatted = formatted.replace(unicode_char, ascii_replacement)
                
            return formatted
            
        except UnicodeEncodeError:
            # Fallback: encode to ASCII with replacement
            try:
                formatted = super().format(record)
                return formatted.encode('ascii', errors='replace').decode('ascii')
            except Exception:
                # Ultimate fallback: basic message
                return f"{record.levelname}: {str(record.msg)}"
        except Exception as e:
            # Handle any other formatting errors
            return f"LOGGING_ERROR: {str(e)} - Original: {str(record.msg)}"


def setup_logging(log_level=logging.INFO):
    """
    Set up logging for the Arena Bot application.
    
    Args:
        log_level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"arena_bot_{timestamp}.log"
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ENHANCED: Robust console handler with multi-strategy UTF-8 setup
    console_handler = None
    encoding_success = False
    
    # Strategy 1: Try StreamHandler with explicit UTF-8 encoding (Python 3.7+)
    try:
        if sys.version_info >= (3, 7):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            
            # Try modern reconfigure method
            if hasattr(console_handler.stream, 'reconfigure'):
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
                encoding_success = True
            elif hasattr(console_handler.stream, 'buffer'):
                # Strategy 2: TextIOWrapper approach for older versions
                console_handler.stream = io.TextIOWrapper(
                    console_handler.stream.buffer, 
                    encoding='utf-8', 
                    errors='replace',
                    newline='\n'
                )
                encoding_success = True
    except (AttributeError, OSError, TypeError) as e:
        # UTF-8 setup failed, will use fallback
        pass
    
    # Strategy 3: Fallback to basic StreamHandler with safe formatter
    if not console_handler or not encoding_success:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
    
    # Use SafeUnicodeFormatter for maximum compatibility
    safe_formatter = SafeUnicodeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(safe_formatter)
    
    # ENHANCED: File handler with robust UTF-8 encoding and error handling
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024, 
            backupCount=5, 
            encoding='utf-8',
            errors='replace'  # Handle encoding errors gracefully
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
    except (OSError, UnicodeError) as e:
        # Fallback: Use basic file handler with ASCII-safe formatter
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024, 
            backupCount=5, 
            encoding='ascii',
            errors='replace'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(safe_formatter)
        print(f"Warning: File logging using ASCII fallback due to: {e}")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {logging.getLevelName(log_level)}")
    logger.info(f"Log file: {log_file}")


def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)