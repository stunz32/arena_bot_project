"""
Formatters for the S-Tier Logging System.

This module contains formatters that convert log records into various output
formats, including structured JSON, human-readable console output, and
space-efficient compressed formats.

Formatters:
- StructuredFormatter: High-performance JSON formatting
- ConsoleFormatter: Human-readable console output with colors
- CompressionFormatter: Space-efficient binary format
"""

from .structured_formatter import StructuredFormatter
from .console_formatter import ConsoleFormatter
from .compression_formatter import CompressionFormatter

__all__ = [
    'StructuredFormatter',
    'ConsoleFormatter', 
    'CompressionFormatter'
]