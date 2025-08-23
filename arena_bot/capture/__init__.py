"""Capture backend package for arena bot."""

from .capture_backend import (
    CaptureBackend,
    CaptureFrame,
    MonitorInfo,
    WindowInfo,
    DXGICaptureBackend,
    BitBltCaptureBackend,
    AdaptiveCaptureManager
)

__all__ = [
    'CaptureBackend',
    'CaptureFrame', 
    'MonitorInfo',
    'WindowInfo',
    'DXGICaptureBackend',
    'BitBltCaptureBackend',
    'AdaptiveCaptureManager'
]