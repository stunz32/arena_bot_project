import sys
from loguru import logger
from pathlib import Path

def setup_logging():
    """
    Set up Loguru for detailed and beautiful logging.
    """
    logger.remove() # Remove default handler
    
    # Console logger with colors and better formatting
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # File logger for history
    log_file = log_dir / "arena_bot_{time:YYYY-MM-DD}.log"
    logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="7 days", encoding="utf-8")
    
    logger.info("Logging configured with Loguru.")

def get_logger(name):
    """No-op, Loguru handles this automatically."""
    return logger