"""
Platform Abstraction Layer for Arena Bot AI Helper v2.

This module provides a comprehensive platform abstraction layer that isolates
all OS-specific code behind clean interfaces, enabling cross-platform compatibility
and easier testing.

Features:
- P0.10.1: Platform Abstraction Layer implementation
- Cross-platform file system operations
- OS-specific window management abstractions
- Platform-aware resource monitoring
- Hardware detection and capabilities
- Network and system information abstraction

Author: Claude (Anthropic)
Created: 2025-07-28
"""

import os
import sys
import platform
import subprocess
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .exceptions import AIHelperPlatformError, AIHelperNotSupportedError


class PlatformType(Enum):
    """Supported platform types"""
    WINDOWS = "windows"
    MACOS = "macos" 
    LINUX = "linux"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Supported architectures"""
    X86 = "x86"
    X64 = "x64"
    ARM64 = "arm64"
    UNKNOWN = "unknown"


@dataclass
class SystemInfo:
    """System information data structure"""
    platform: PlatformType
    architecture: Architecture
    os_version: str
    python_version: str
    display_scaling: float = 1.0
    monitor_count: int = 1
    total_memory_gb: float = 0.0
    cpu_cores: int = 1
    gpu_info: Optional[str] = None


class FileSystemAbstraction(ABC):
    """Abstract interface for file system operations"""
    
    @abstractmethod
    def get_home_directory(self) -> Path:
        """Get user home directory"""
        pass
    
    @abstractmethod
    def get_app_data_directory(self, app_name: str) -> Path:
        """Get application data directory"""
        pass
    
    @abstractmethod
    def get_temp_directory(self) -> Path:
        """Get temporary directory"""
        pass
    
    @abstractmethod
    def get_executable_directory(self) -> Path:
        """Get directory containing current executable"""
        pass
    
    @abstractmethod
    def is_path_writable(self, path: Path) -> bool:
        """Check if path is writable"""
        pass
    
    @abstractmethod
    def create_directory_safe(self, path: Path) -> bool:
        """Safely create directory with proper permissions"""
        pass
    
    @abstractmethod
    def get_file_permissions(self, path: Path) -> str:
        """Get file permissions as string"""
        pass
    
    @abstractmethod
    def set_file_permissions(self, path: Path, permissions: str) -> bool:
        """Set file permissions"""
        pass


class WindowManagementAbstraction(ABC):
    """Abstract interface for window management operations"""
    
    @abstractmethod
    def get_active_window_title(self) -> Optional[str]:
        """Get title of currently active window"""
        pass
    
    @abstractmethod
    def get_window_bounds(self, window_title: str) -> Optional[Tuple[int, int, int, int]]:
        """Get window bounds (x, y, width, height)"""
        pass
    
    @abstractmethod
    def is_window_visible(self, window_title: str) -> bool:
        """Check if window is visible"""
        pass
    
    @abstractmethod
    def bring_window_to_front(self, window_title: str) -> bool:
        """Bring window to front"""
        pass
    
    @abstractmethod
    def get_screen_resolution(self) -> Tuple[int, int]:
        """Get primary screen resolution"""
        pass
    
    @abstractmethod
    def get_all_screen_resolutions(self) -> List[Tuple[int, int]]:
        """Get all screen resolutions"""
        pass
    
    @abstractmethod
    def get_dpi_scaling(self) -> float:
        """Get display scaling factor"""
        pass


class ProcessManagementAbstraction(ABC):
    """Abstract interface for process management"""
    
    @abstractmethod
    def get_process_list(self) -> List[Dict[str, Any]]:
        """Get list of running processes"""
        pass
    
    @abstractmethod
    def is_process_running(self, process_name: str) -> bool:
        """Check if process is running"""
        pass
    
    @abstractmethod
    def get_process_memory_usage(self, pid: int) -> float:
        """Get process memory usage in MB"""
        pass
    
    @abstractmethod
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information"""
        pass
    
    @abstractmethod
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        pass
    
    @abstractmethod
    def kill_process(self, pid: int) -> bool:
        """Terminate process by PID"""
        pass


class NetworkAbstraction(ABC):
    """Abstract interface for network operations"""
    
    @abstractmethod
    def is_internet_available(self) -> bool:
        """Check if internet connection is available"""
        pass
    
    @abstractmethod
    def get_local_ip(self) -> Optional[str]:
        """Get local IP address"""
        pass
    
    @abstractmethod
    def ping_host(self, hostname: str, timeout: float = 5.0) -> bool:
        """Ping a host to check connectivity"""
        pass
    
    @abstractmethod
    def get_network_interfaces(self) -> List[Dict[str, str]]:
        """Get network interface information"""
        pass


# Windows Implementation
class WindowsFileSystem(FileSystemAbstraction):
    """Windows-specific file system implementation"""
    
    def get_home_directory(self) -> Path:
        return Path.home()
    
    def get_app_data_directory(self, app_name: str) -> Path:
        app_data = os.environ.get('APPDATA', str(Path.home() / 'AppData' / 'Roaming'))
        return Path(app_data) / app_name
    
    def get_temp_directory(self) -> Path:
        return Path(os.environ.get('TEMP', 'C:/temp'))
    
    def get_executable_directory(self) -> Path:
        return Path(sys.executable).parent
    
    def is_path_writable(self, path: Path) -> bool:
        try:
            test_file = path / '.write_test'
            test_file.touch()
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False
    
    def create_directory_safe(self, path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except (OSError, PermissionError):
            return False
    
    def get_file_permissions(self, path: Path) -> str:
        try:
            import stat
            mode = path.stat().st_mode
            return stat.filemode(mode)
        except (OSError, ImportError):
            return "unknown"
    
    def set_file_permissions(self, path: Path, permissions: str) -> bool:
        try:
            # Windows has limited POSIX permission support
            if 'w' not in permissions:
                path.chmod(0o444)  # Read-only
            else:
                path.chmod(0o666)  # Read-write
            return True
        except (OSError, ValueError):
            return False


class WindowsWindowManagement(WindowManagementAbstraction):
    """Windows-specific window management implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".windows")
        self._win32_available = False
        try:
            import win32gui
            import win32api
            import win32con
            self._win32gui = win32gui
            self._win32api = win32api
            self._win32con = win32con
            self._win32_available = True
        except ImportError:
            self.logger.warning("win32 modules not available, window management will be limited")
    
    def get_active_window_title(self) -> Optional[str]:
        if not self._win32_available:
            return None
        
        try:
            hwnd = self._win32gui.GetForegroundWindow()
            return self._win32gui.GetWindowText(hwnd)
        except Exception as e:
            self.logger.error(f"Failed to get active window title: {e}")
            return None
    
    def get_window_bounds(self, window_title: str) -> Optional[Tuple[int, int, int, int]]:
        if not self._win32_available:
            return None
        
        try:
            def callback(hwnd, windows):
                if self._win32gui.IsWindowVisible(hwnd) and window_title.lower() in self._win32gui.GetWindowText(hwnd).lower():
                    windows.append(hwnd)
                return True
            
            windows = []
            self._win32gui.EnumWindows(callback, windows)
            
            if windows:
                rect = self._win32gui.GetWindowRect(windows[0])
                return rect  # (left, top, right, bottom)
            
        except Exception as e:
            self.logger.error(f"Failed to get window bounds: {e}")
        
        return None
    
    def is_window_visible(self, window_title: str) -> bool:
        bounds = self.get_window_bounds(window_title)
        return bounds is not None
    
    def bring_window_to_front(self, window_title: str) -> bool:
        if not self._win32_available:
            return False
        
        try:
            def callback(hwnd, windows):
                if self._win32gui.IsWindowVisible(hwnd) and window_title.lower() in self._win32gui.GetWindowText(hwnd).lower():
                    windows.append(hwnd)
                return True
            
            windows = []
            self._win32gui.EnumWindows(callback, windows)
            
            if windows:
                self._win32gui.SetForegroundWindow(windows[0])
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to bring window to front: {e}")
        
        return False
    
    def get_screen_resolution(self) -> Tuple[int, int]:
        if self._win32_available:
            try:
                return (
                    self._win32api.GetSystemMetrics(self._win32con.SM_CXSCREEN),
                    self._win32api.GetSystemMetrics(self._win32con.SM_CYSCREEN)
                )
            except Exception:
                pass
        
        # Fallback
        return (1920, 1080)
    
    def get_all_screen_resolutions(self) -> List[Tuple[int, int]]:
        # For now, return primary screen
        # Could be enhanced with multi-monitor support
        return [self.get_screen_resolution()]
    
    def get_dpi_scaling(self) -> float:
        if self._win32_available:
            try:
                import win32print
                hdc = self._win32gui.GetDC(0)
                dpi = self._win32gui.GetDeviceCaps(hdc, win32con.LOGPIXELSX)
                self._win32gui.ReleaseDC(0, hdc)
                return dpi / 96.0  # 96 DPI is 100% scaling
            except Exception:
                pass
        
        return 1.0


class WindowsProcessManagement(ProcessManagementAbstraction):
    """Windows-specific process management implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".windows_process")
        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            self.logger.warning("psutil not available, process management will be limited")
            self._psutil_available = False
    
    def get_process_list(self) -> List[Dict[str, Any]]:
        if not self._psutil_available:
            return []
        
        try:
            processes = []
            for proc in self._psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    processes.append(proc.info)
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    continue
            return processes
        except Exception as e:
            self.logger.error(f"Failed to get process list: {e}")
            return []
    
    def is_process_running(self, process_name: str) -> bool:
        if not self._psutil_available:
            return False
        
        try:
            for proc in self._psutil.process_iter(['name']):
                try:
                    if proc.info['name'].lower() == process_name.lower():
                        return True
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    continue
            return False
        except Exception as e:
            self.logger.error(f"Failed to check if process is running: {e}")
            return False
    
    def get_process_memory_usage(self, pid: int) -> float:
        if not self._psutil_available:
            return 0.0
        
        try:
            proc = self._psutil.Process(pid)
            return proc.memory_info().rss / 1024 / 1024  # MB
        except Exception as e:
            self.logger.error(f"Failed to get process memory usage: {e}")
            return 0.0
    
    def get_system_memory_info(self) -> Dict[str, float]:
        if not self._psutil_available:
            return {'total': 0, 'available': 0, 'used': 0, 'percentage': 0}
        
        try:
            memory = self._psutil.virtual_memory()
            return {
                'total': memory.total / 1024 / 1024 / 1024,  # GB
                'available': memory.available / 1024 / 1024 / 1024,  # GB
                'used': memory.used / 1024 / 1024 / 1024,  # GB
                'percentage': memory.percent
            }
        except Exception as e:
            self.logger.error(f"Failed to get system memory info: {e}")
            return {'total': 0, 'available': 0, 'used': 0, 'percentage': 0}
    
    def get_cpu_usage(self) -> float:
        if not self._psutil_available:
            return 0.0
        
        try:
            return self._psutil.cpu_percent(interval=0.1)
        except Exception as e:
            self.logger.error(f"Failed to get CPU usage: {e}")
            return 0.0
    
    def kill_process(self, pid: int) -> bool:
        if not self._psutil_available:
            return False
        
        try:
            proc = self._psutil.Process(pid)
            proc.terminate()
            return True
        except Exception as e:
            self.logger.error(f"Failed to kill process: {e}")
            return False


class WindowsNetworkAbstraction(NetworkAbstraction):
    """Windows-specific network implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".windows_network")
    
    def is_internet_available(self) -> bool:
        return self.ping_host("8.8.8.8", 3.0)
    
    def get_local_ip(self) -> Optional[str]:
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return None
    
    def ping_host(self, hostname: str, timeout: float = 5.0) -> bool:
        try:
            result = subprocess.run(
                ['ping', '-n', '1', '-w', str(int(timeout * 1000)), hostname],
                capture_output=True,
                timeout=timeout + 1
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_network_interfaces(self) -> List[Dict[str, str]]:
        try:
            import psutil
            interfaces = []
            for interface_name, addresses in psutil.net_if_addrs().items():
                for address in addresses:
                    if address.family == 2:  # AF_INET
                        interfaces.append({
                            'name': interface_name,
                            'ip': address.address,
                            'netmask': address.netmask
                        })
            return interfaces
        except Exception as e:
            self.logger.error(f"Failed to get network interfaces: {e}")
            return []


# Platform Detection and Factory
class PlatformAbstractionLayer:
    """
    P0.10.1: Main Platform Abstraction Layer
    
    Provides unified access to platform-specific implementations
    through abstract interfaces.
    """
    
    def __init__(self):
        """Initialize platform abstraction layer"""
        self.logger = logging.getLogger(__name__)
        
        # Detect platform
        self._platform_type = self._detect_platform()
        self._architecture = self._detect_architecture()
        self._system_info = self._gather_system_info()
        
        # Initialize platform-specific implementations
        self._file_system = self._create_file_system()
        self._window_management = self._create_window_management()
        self._process_management = self._create_process_management()
        self._network = self._create_network()
        
        self.logger.info(f"Platform abstraction layer initialized for {self._platform_type.value}")
    
    def _detect_platform(self) -> PlatformType:
        """Detect current platform"""
        system = platform.system().lower()
        
        if system == 'windows':
            return PlatformType.WINDOWS
        elif system == 'darwin':
            return PlatformType.MACOS
        elif system == 'linux':
            return PlatformType.LINUX
        else:
            return PlatformType.UNKNOWN
    
    def _detect_architecture(self) -> Architecture:
        """Detect system architecture"""
        arch = platform.machine().lower()
        
        if arch in ('x86_64', 'amd64'):
            return Architecture.X64
        elif arch in ('i386', 'i686', 'x86'):
            return Architecture.X86
        elif arch in ('arm64', 'aarch64'):
            return Architecture.ARM64
        else:
            return Architecture.UNKNOWN
    
    def _gather_system_info(self) -> SystemInfo:
        """Gather comprehensive system information"""
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
            cpu_cores = psutil.cpu_count()
        except ImportError:
            total_memory = 0.0
            cpu_cores = os.cpu_count() or 1
        
        return SystemInfo(
            platform=self._platform_type,
            architecture=self._architecture,
            os_version=platform.version(),
            python_version=platform.python_version(),
            total_memory_gb=total_memory,
            cpu_cores=cpu_cores
        )
    
    def _create_file_system(self) -> FileSystemAbstraction:
        """Create platform-specific file system implementation"""
        if self._platform_type == PlatformType.WINDOWS:
            return WindowsFileSystem()
        else:
            # For now, use Windows implementation as fallback
            # TODO: Implement MacOS and Linux specific implementations
            return WindowsFileSystem()
    
    def _create_window_management(self) -> WindowManagementAbstraction:
        """Create platform-specific window management implementation"""
        if self._platform_type == PlatformType.WINDOWS:
            return WindowsWindowManagement()
        else:
            # TODO: Implement MacOS and Linux specific implementations
            return WindowsWindowManagement()
    
    def _create_process_management(self) -> ProcessManagementAbstraction:
        """Create platform-specific process management implementation"""
        if self._platform_type == PlatformType.WINDOWS:
            return WindowsProcessManagement()
        else:
            # TODO: Implement MacOS and Linux specific implementations
            return WindowsProcessManagement()
    
    def _create_network(self) -> NetworkAbstraction:
        """Create platform-specific network implementation"""
        if self._platform_type == PlatformType.WINDOWS:
            return WindowsNetworkAbstraction()
        else:
            # TODO: Implement MacOS and Linux specific implementations
            return WindowsNetworkAbstraction()
    
    @property
    def platform_type(self) -> PlatformType:
        """Get current platform type"""
        return self._platform_type
    
    @property
    def architecture(self) -> Architecture:
        """Get system architecture"""
        return self._architecture
    
    @property
    def system_info(self) -> SystemInfo:
        """Get comprehensive system information"""
        return self._system_info
    
    @property
    def file_system(self) -> FileSystemAbstraction:
        """Get file system abstraction"""
        return self._file_system
    
    @property
    def window_management(self) -> WindowManagementAbstraction:
        """Get window management abstraction"""
        return self._window_management
    
    @property
    def process_management(self) -> ProcessManagementAbstraction:
        """Get process management abstraction"""
        return self._process_management
    
    @property
    def network(self) -> NetworkAbstraction:
        """Get network abstraction"""
        return self._network
    
    def is_supported_platform(self) -> bool:
        """Check if current platform is fully supported"""
        return self._platform_type != PlatformType.UNKNOWN
    
    def get_compatibility_matrix(self) -> Dict[str, bool]:
        """Get platform compatibility matrix"""
        return {
            'file_system': True,
            'window_management': self._platform_type == PlatformType.WINDOWS,
            'process_management': True,
            'network': True,
            'full_support': self._platform_type == PlatformType.WINDOWS
        }
    
    def validate_environment(self) -> List[str]:
        """Validate current environment and return list of issues"""
        issues = []
        
        if not self.is_supported_platform():
            issues.append(f"Platform {self._platform_type.value} is not fully supported")
        
        if self._system_info.total_memory_gb < 2:
            issues.append("System has less than 2GB RAM, performance may be degraded")
        
        if self._system_info.cpu_cores < 2:
            issues.append("System has only 1 CPU core, performance may be limited")
        
        # Check for required directories
        app_data_dir = self._file_system.get_app_data_directory("ArenaBot")
        if not self._file_system.create_directory_safe(app_data_dir):
            issues.append(f"Cannot create application data directory: {app_data_dir}")
        
        return issues


# Global instance
_platform_layer = None

def get_platform_layer() -> PlatformAbstractionLayer:
    """Get global platform abstraction layer instance"""
    global _platform_layer
    if _platform_layer is None:
        _platform_layer = PlatformAbstractionLayer()
    return _platform_layer


# Convenience functions
def get_system_info() -> SystemInfo:
    """Get system information"""
    return get_platform_layer().system_info


def is_windows() -> bool:
    """Check if running on Windows"""
    return get_platform_layer().platform_type == PlatformType.WINDOWS


def is_macos() -> bool:
    """Check if running on macOS"""
    return get_platform_layer().platform_type == PlatformType.MACOS


def is_linux() -> bool:
    """Check if running on Linux"""
    return get_platform_layer().platform_type == PlatformType.LINUX


def get_platform_manager() -> PlatformAbstractionLayer:
    """
    Get the platform manager instance (alias for get_platform_layer).
    
    This function provides compatibility with the expected import name
    used by other components in the AI v2 system.
    
    Returns:
        PlatformAbstractionLayer: The global platform abstraction layer instance
    """
    return get_platform_layer()


# Export main components
__all__ = [
    # Core Classes
    'PlatformAbstractionLayer',
    
    # Enums
    'PlatformType',
    'Architecture',
    
    # Data Classes
    'SystemInfo',
    
    # Abstract Interfaces
    'FileSystemAbstraction',
    'WindowManagementAbstraction', 
    'ProcessManagementAbstraction',
    'NetworkAbstraction',
    
    # Factory Functions
    'get_platform_layer',
    'get_platform_manager',  # Added missing function
    'get_system_info',
    
    # Convenience Functions
    'is_windows',
    'is_macos',
    'is_linux'
]