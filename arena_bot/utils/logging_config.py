"""
Simple logging configuration for Arena Bot.

Following CLAUDE.md principles - minimal, focused, and easy to understand.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


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
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler (with rotation)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
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