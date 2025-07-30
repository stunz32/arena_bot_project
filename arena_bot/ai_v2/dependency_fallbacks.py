"""
AI Helper v2 - Dependency Fallback System
Provides graceful degradation when optional dependencies are missing

This module implements intelligent fallbacks that allow the system to continue
functioning with reduced capabilities when optional dependencies are unavailable.
"""

import logging
import sys
import warnings
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
import uuid

# Set up correlation ID logging
class CorrelationIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = str(uuid.uuid4())[:8]
        return True

logger = logging.getLogger(__name__)
logger.addFilter(CorrelationIDFilter())

class DependencyFallback:
    """Manages fallback implementations for missing dependencies"""
    
    def __init__(self):
        self._available_deps = {}
        self._fallback_registry = {}
        self._capability_cache = {}
        
    def register_fallback(self, package_name: str, fallback_impl: Callable):
        """Register a fallback implementation for a package"""
        self._fallback_registry[package_name] = fallback_impl
        logger.info(f"Registered fallback for {package_name}")
        
    def is_available(self, package_name: str) -> bool:
        """Check if a package is available"""
        if package_name in self._available_deps:
            return self._available_deps[package_name]
            
        try:
            __import__(package_name)
            self._available_deps[package_name] = True
            logger.debug(f"{package_name} is available")
            return True
        except ImportError:
            self._available_deps[package_name] = False
            logger.info(f"{package_name} not available - fallback may be used")
            return False
            
    def get_implementation(self, package_name: str, capability: str = None):
        """Get either the real package or fallback implementation"""
        if self.is_available(package_name):
            return __import__(package_name)
        elif package_name in self._fallback_registry:
            logger.info(f"Using fallback implementation for {package_name}")
            return self._fallback_registry[package_name]
        else:
            raise ImportError(f"No fallback available for {package_name}")

# Global fallback manager
fallback_manager = DependencyFallback()

# === Data Validation Fallbacks ===

class ManualDataValidator:
    """Manual data validation fallback for pydantic"""
    
    @staticmethod
    def validate_dict(data: dict, schema: dict) -> dict:
        """Basic dictionary validation"""
        validated = {}
        
        for key, expected_type in schema.items():
            if key in data:
                value = data[key]
                if isinstance(expected_type, type):
                    if not isinstance(value, expected_type):
                        try:
                            validated[key] = expected_type(value)
                        except (ValueError, TypeError):
                            raise ValueError(f"Cannot convert {key} to {expected_type.__name__}")
                    else:
                        validated[key] = value
                else:
                    # More complex validation would go here
                    validated[key] = value
            else:
                raise ValueError(f"Required field {key} missing")
                
        return validated
        
    @staticmethod
    def validate_card_data(card_data: dict) -> dict:
        """Validate card data structure"""
        schema = {
            'name': str,
            'cost': int,
            'attack': int,
            'health': int,
            'card_class': str,
            'rarity': str
        }
        
        return ManualDataValidator.validate_dict(card_data, schema)

fallback_manager.register_fallback('pydantic', ManualDataValidator)

# === Resource Monitoring Fallbacks ===

class BasicResourceMonitor:
    """Basic resource monitoring fallback for psutil"""
    
    def __init__(self):
        self._process_count = 0
        
    def cpu_percent(self, interval=None):
        """Mock CPU percentage - returns reasonable values"""
        import time
        import os
        
        # Use process count as a rough CPU indicator
        try:
            import subprocess
            result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
            process_count = len(result.stdout.split('\n'))
            
            # Normalize to 0-100 range
            cpu_estimate = min(100, max(0, (process_count - 50) * 2))
            return cpu_estimate
        except:
            return 25.0  # Default moderate usage
            
    def virtual_memory(self):
        """Mock memory information"""
        class MemoryInfo:
            def __init__(self):
                # Estimate based on system architecture
                if sys.maxsize > 2**32:  # 64-bit system
                    self.total = 8 * 1024 * 1024 * 1024  # 8GB estimate
                    self.available = 4 * 1024 * 1024 * 1024  # 4GB estimate
                    self.used = self.total - self.available
                else:  # 32-bit system
                    self.total = 4 * 1024 * 1024 * 1024  # 4GB max
                    self.available = 2 * 1024 * 1024 * 1024  # 2GB estimate
                    self.used = self.total - self.available
                    
                self.percent = (self.used / self.total) * 100
                
        return MemoryInfo()
        
    def disk_usage(self, path='/'):
        """Mock disk usage information"""
        class DiskInfo:
            def __init__(self):
                # Conservative estimates
                self.total = 500 * 1024 * 1024 * 1024  # 500GB
                self.used = 250 * 1024 * 1024 * 1024   # 250GB
                self.free = self.total - self.used
                
        return DiskInfo()
        
    def Process(self, pid=None):
        """Mock process information"""
        class ProcessInfo:
            def __init__(self, pid):
                self.pid = pid or os.getpid()
                
            def memory_info(self):
                class MemInfo:
                    def __init__(self):
                        self.rss = 100 * 1024 * 1024  # 100MB estimate
                        self.vms = 200 * 1024 * 1024  # 200MB estimate
                return MemInfo()
                
            def cpu_percent(self):
                return 5.0  # 5% estimate
                
        return ProcessInfo(pid)

fallback_manager.register_fallback('psutil', BasicResourceMonitor())

# === Configuration Management Fallbacks ===

class BasicConfigManager:
    """Basic configuration management fallback"""
    
    def __init__(self):
        self._config = {}
        
    def load_yaml(self, file_path: str) -> dict:
        """Basic YAML-like configuration loading"""
        config = {}
        try:
            with open(file_path, 'r') as f:
                current_section = config
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    if ':' in line and not line.startswith(' '):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Basic type conversion
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace('.', '').isdigit():
                            value = float(value)
                        elif value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                            
                        current_section[key] = value
                        
        except Exception as e:
            logger.warning(f"Failed to load config file {file_path}: {e}")
            
        return config
        
    def validate_schema(self, data: dict, schema: dict) -> bool:
        """Basic schema validation"""
        for key, expected_type in schema.items():
            if key not in data:
                logger.error(f"Missing required config key: {key}")
                return False
            if not isinstance(data[key], expected_type):
                logger.error(f"Config key {key} has wrong type: expected {expected_type}, got {type(data[key])}")
                return False
        return True

fallback_manager.register_fallback('yaml', BasicConfigManager())
fallback_manager.register_fallback('jsonschema', BasicConfigManager())

# === Machine Learning Fallbacks ===

class BasicMLFallback:
    """Basic ML operations fallback"""
    
    def __init__(self):
        self.models = {}
        
    def simple_heuristic_scorer(self, card_data: dict) -> float:
        """Simple heuristic scoring when ML models fail"""
        try:
            # Basic card value formula: cost efficiency + tempo
            cost = card_data.get('cost', 1)
            attack = card_data.get('attack', 0)
            health = card_data.get('health', 0)
            
            if cost == 0:
                cost = 1  # Avoid division by zero
                
            # Basic value formula
            stats_value = attack + health
            efficiency = stats_value / cost
            
            # Apply rarity bonus
            rarity_bonus = {
                'common': 0,
                'rare': 0.2,
                'epic': 0.4,
                'legendary': 0.6
            }.get(card_data.get('rarity', 'common').lower(), 0)
            
            base_score = efficiency + rarity_bonus
            
            # Normalize to 0-100 scale
            normalized_score = min(100, max(0, base_score * 25))
            
            return normalized_score
            
        except Exception as e:
            logger.warning(f"Heuristic scoring failed: {e}")
            return 50.0  # Default moderate score

fallback_manager.register_fallback('lightgbm', BasicMLFallback())

# === Caching Fallbacks ===

class BasicCache:
    """Basic in-memory cache fallback"""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache = {}
        self._access_order = []
        
    def get(self, key: str, default=None):
        """Get value from cache"""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return default
        
    def set(self, key: str, value: Any):
        """Set value in cache"""
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new
            if len(self._cache) >= self.maxsize:
                # Remove least recently used
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
                
            self._cache[key] = value
            self._access_order.append(key)
            
    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self._access_order.clear()

fallback_manager.register_fallback('lru-dict', BasicCache)

# === Performance Profiling Fallbacks ===

class BasicProfiler:
    """Basic profiling fallback"""
    
    def __init__(self):
        self.start_time = None
        
    def profile(self, func):
        """Basic function profiling decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                logger.info(f"Function {func.__name__} took {end_time - start_time:.3f}s")
                return result
            except Exception as e:
                end_time = time.time()
                logger.error(f"Function {func.__name__} failed after {end_time - start_time:.3f}s: {e}")
                raise
        return wrapper
        
    def memory_usage(self, func):
        """Mock memory usage profiling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Memory profiling not available for {func.__name__} - using fallback")
            return func(*args, **kwargs)
        return wrapper

fallback_manager.register_fallback('memory-profiler', BasicProfiler())
fallback_manager.register_fallback('line-profiler', BasicProfiler())

# === Utility Functions ===

def safe_import(package_name: str, capability: str = None):
    """Safely import a package with fallback support"""
    try:
        return fallback_manager.get_implementation(package_name, capability)
    except ImportError as e:
        logger.error(f"No implementation available for {package_name}: {e}")
        raise

def require_fallback(packages: List[str]) -> Dict[str, Any]:
    """Require packages with fallback support"""
    implementations = {}
    
    for package in packages:
        try:
            implementations[package] = safe_import(package)
        except ImportError:
            logger.warning(f"Package {package} not available and no fallback registered")
            implementations[package] = None
            
    return implementations

def capability_available(capability: str) -> bool:
    """Check if a capability is available (either native or fallback)"""
    capability_map = {
        'data_validation': ['pydantic'],
        'resource_monitoring': ['psutil'],
        'yaml_config': ['yaml'],
        'schema_validation': ['jsonschema'],
        'ml_models': ['lightgbm', 'scikit-learn'],
        'caching': ['lru-dict'],
        'profiling': ['memory-profiler', 'line-profiler']
    }
    
    if capability not in capability_map:
        return False
        
    packages = capability_map[capability]
    
    for package in packages:
        if fallback_manager.is_available(package) or package in fallback_manager._fallback_registry:
            return True
            
    return False

def get_capability_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available capabilities"""
    capabilities = {
        'data_validation': {
            'available': capability_available('data_validation'),
            'native': fallback_manager.is_available('pydantic'),
            'fallback': 'pydantic' in fallback_manager._fallback_registry
        },
        'resource_monitoring': {
            'available': capability_available('resource_monitoring'),
            'native': fallback_manager.is_available('psutil'),
            'fallback': 'psutil' in fallback_manager._fallback_registry
        },
        'configuration': {
            'available': capability_available('yaml_config'),
            'native': fallback_manager.is_available('yaml'),
            'fallback': 'yaml' in fallback_manager._fallback_registry
        },
        'machine_learning': {
            'available': capability_available('ml_models'),
            'native': fallback_manager.is_available('lightgbm'),
            'fallback': 'lightgbm' in fallback_manager._fallback_registry
        },
        'caching': {
            'available': capability_available('caching'),
            'native': fallback_manager.is_available('lru-dict'),
            'fallback': 'lru-dict' in fallback_manager._fallback_registry
        }
    }
    
    return capabilities

# Initialize fallbacks on import
def _initialize_fallbacks():
    """Initialize fallback system"""
    logger.info("Initializing dependency fallback system")
    
    # Check availability of key packages
    key_packages = ['pydantic', 'psutil', 'yaml', 'jsonschema', 'lightgbm', 'lru-dict']
    
    available_count = 0
    for package in key_packages:
        if fallback_manager.is_available(package):
            available_count += 1
            
    fallback_count = len(fallback_manager._fallback_registry)
    
    logger.info(f"Dependency fallback system initialized: {available_count}/{len(key_packages)} native packages available, {fallback_count} fallbacks registered")
    
    if available_count < len(key_packages):
        logger.info("System will operate with reduced functionality due to missing optional dependencies")

# Initialize on import
_initialize_fallbacks()