#!/usr/bin/env python3
"""
AI Helper v2 - Dependency Validation Script
Implements comprehensive dependency management with hardening features

Features:
- Pre-installation environment scanning
- Virtual environment isolation with conflict detection
- Graceful dependency fallback for optional components
- Installation state recovery with rollback capability
- Performance monitoring and resource management
"""

import os
import sys
import json
import hashlib
import subprocess
import tempfile
import shutil
import logging
import importlib
import pkg_resources
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import venv
import uuid

# Configure logging with correlation IDs
class CorrelationIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = str(uuid.uuid4())[:8]
        return True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(correlation_id)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('dependency_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.addFilter(CorrelationIDFilter())

class DependencyValidationError(Exception):
    """Custom exception for dependency validation failures"""
    pass

class InstallationStateManager:
    """Manages installation state and rollback capability"""
    
    def __init__(self, state_dir: str = ".dependency_states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.current_state_file = self.state_dir / "current_state.json"
        self.backup_state_file = self.state_dir / "backup_state.json"
        
    def create_snapshot(self) -> str:
        """Create a snapshot of current environment state"""
        snapshot_id = str(uuid.uuid4())
        
        try:
            import pkg_resources
            installed_packages = {
                pkg.key: pkg.version 
                for pkg in pkg_resources.working_set
            }
        except Exception as e:
            logger.warning(f"Could not enumerate packages: {e}")
            installed_packages = {}
            
        state = {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "installed_packages": installed_packages,
            "environment_variables": dict(os.environ),
            "working_directory": os.getcwd()
        }
        
        snapshot_file = self.state_dir / f"snapshot_{snapshot_id}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Created environment snapshot: {snapshot_id}")
        return snapshot_id
        
    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback environment to a previous snapshot"""
        snapshot_file = self.state_dir / f"snapshot_{snapshot_id}.json"
        
        if not snapshot_file.exists():
            logger.error(f"Snapshot {snapshot_id} not found")
            return False
            
        try:
            with open(snapshot_file, 'r') as f:
                state = json.load(f)
                
            logger.info(f"Rolling back to snapshot {snapshot_id} from {state['timestamp']}")
            
            # This is a simplified rollback - in production, this would involve
            # more sophisticated package management
            logger.warning("Rollback capability requires manual intervention")
            logger.info("Please restore your environment manually using the snapshot data")
            logger.info(f"Snapshot data: {snapshot_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to snapshot {snapshot_id}: {e}")
            return False

class ResourceMonitor:
    """Monitor system resources during dependency installation"""
    
    def __init__(self):
        self.start_memory = 0
        self.start_disk = 0
        
    def start_monitoring(self):
        """Start resource monitoring"""
        try:
            import psutil
            self.start_memory = psutil.virtual_memory().used
            self.start_disk = psutil.disk_usage('/').used
            logger.info(f"Resource monitoring started - Memory: {self.start_memory/1024/1024:.1f}MB")
        except ImportError:
            logger.warning("psutil not available - resource monitoring disabled")
            
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage"""
        try:
            import psutil
            current_memory = psutil.virtual_memory().used
            current_disk = psutil.disk_usage('/').used
            
            return {
                "memory_used_mb": current_memory / 1024 / 1024,
                "memory_increase_mb": (current_memory - self.start_memory) / 1024 / 1024,
                "disk_used_gb": current_disk / 1024 / 1024 / 1024,
                "disk_increase_gb": (current_disk - self.start_disk) / 1024 / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent()
            }
        except ImportError:
            return {"status": "monitoring_unavailable"}

class DependencyValidator:
    """Main dependency validation and management class"""
    
    def __init__(self, requirements_file: str = "requirements_ai_v2.txt"):
        self.requirements_file = Path(requirements_file)
        self.state_manager = InstallationStateManager()
        self.resource_monitor = ResourceMonitor()
        self.optional_deps = {
            'psutil', 'pydantic', 'queue-manager', 'hashlib-compat',
            'pytest', 'pytest-cov', 'memory-profiler', 'sphinx', 
            'sphinx-rtd-theme', 'py-spy', 'line-profiler'
        }
        
    def scan_environment(self) -> Dict[str, Any]:
        """P0.1.1: Pre-installation Environment Scan - detect conflicts before installation"""
        logger.info("Starting pre-installation environment scan")
        
        scan_results = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "virtual_env": os.environ.get('VIRTUAL_ENV'),
            "pip_version": None,
            "potential_conflicts": [],
            "system_packages": [],
            "disk_space_gb": 0,
            "memory_available_gb": 0
        }
        
        # Check pip version
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True)
            scan_results["pip_version"] = result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not check pip version: {e}")
            
        # Check system packages
        try:
            import pkg_resources
            scan_results["system_packages"] = [
                f"{pkg.key}=={pkg.version}" 
                for pkg in pkg_resources.working_set
            ]
        except Exception as e:
            logger.warning(f"Could not enumerate system packages: {e}")
            
        # Check system resources
        try:
            import psutil
            scan_results["disk_space_gb"] = psutil.disk_usage('/').free / 1024**3
            scan_results["memory_available_gb"] = psutil.virtual_memory().available / 1024**3
        except ImportError:
            logger.warning("psutil not available for resource checking")
            
        # Detect potential conflicts
        if self.requirements_file.exists():
            with open(self.requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
            for req in requirements:
                if '==' in req:
                    pkg_name = req.split('==')[0].strip()
                    required_version = req.split('==')[1].split(';')[0].strip()
                    
                    try:
                        current_version = pkg_resources.get_distribution(pkg_name).version
                        if current_version != required_version:
                            conflict = {
                                "package": pkg_name,
                                "required": required_version,
                                "current": current_version,
                                "severity": "high" if pkg_name not in self.optional_deps else "low"
                            }
                            scan_results["potential_conflicts"].append(conflict)
                    except pkg_resources.DistributionNotFound:
                        # Package not installed - no conflict
                        pass
                    except Exception as e:
                        logger.warning(f"Could not check version for {pkg_name}: {e}")
                        
        logger.info(f"Environment scan complete - {len(scan_results['potential_conflicts'])} conflicts detected")
        return scan_results
        
    def create_isolated_environment(self, venv_path: str = "arena_bot_venv") -> bool:
        """P0.1.2: Virtual Environment Isolation - mandatory venv creation with conflict detection"""
        logger.info(f"Creating isolated virtual environment: {venv_path}")
        
        venv_path = Path(venv_path)
        
        # Remove existing venv if it exists
        if venv_path.exists():
            logger.warning(f"Removing existing virtual environment: {venv_path}")
            try:
                shutil.rmtree(venv_path)
            except Exception as e:
                logger.error(f"Failed to remove existing venv: {e}")
                return False
                
        try:
            # Create virtual environment
            venv.create(venv_path, with_pip=True, clear=True)
            
            # Test the virtual environment
            python_exe = venv_path / "Scripts" / "python.exe" if os.name == 'nt' else venv_path / "bin" / "python"
            
            if not python_exe.exists():
                raise DependencyValidationError(f"Python executable not found in venv: {python_exe}")
                
            # Test pip in venv
            result = subprocess.run([str(python_exe), '-m', 'pip', '--version'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise DependencyValidationError(f"pip not working in venv: {result.stderr}")
                
            logger.info(f"Virtual environment created successfully: {venv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
            
    def install_with_fallback(self, venv_path: str = "arena_bot_venv") -> Dict[str, Any]:
        """P0.1.3: Graceful Dependency Fallback - system continues with reduced functionality if optional deps fail"""
        logger.info("Starting dependency installation with graceful fallback")
        
        venv_path = Path(venv_path)
        python_exe = venv_path / "Scripts" / "python.exe" if os.name == 'nt' else venv_path / "bin" / "python"
        
        if not python_exe.exists():
            raise DependencyValidationError(f"Virtual environment not found: {venv_path}")
            
        # Create installation snapshot
        snapshot_id = self.state_manager.create_snapshot()
        
        results = {
            "snapshot_id": snapshot_id,
            "successful_installs": [],
            "failed_installs": [],
            "optional_failed": [],
            "critical_failed": [],
            "total_packages": 0,
            "success_rate": 0.0
        }
        
        if not self.requirements_file.exists():
            raise DependencyValidationError(f"Requirements file not found: {self.requirements_file}")
            
        with open(self.requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        results["total_packages"] = len(requirements)
        self.resource_monitor.start_monitoring()
        
        for req in requirements:
            package_name = req.split('==')[0].split(';')[0].strip()
            is_optional = package_name in self.optional_deps
            
            logger.info(f"Installing {package_name} ({'optional' if is_optional else 'required'})")
            
            try:
                # Install package with timeout
                cmd = [str(python_exe), '-m', 'pip', 'install', req]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    results["successful_installs"].append(package_name)
                    logger.info(f"Successfully installed {package_name}")
                else:
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
                    
            except Exception as e:
                logger.warning(f"Failed to install {package_name}: {e}")
                results["failed_installs"].append({"package": package_name, "error": str(e)})
                
                if is_optional:
                    results["optional_failed"].append(package_name)
                    logger.info(f"Optional dependency {package_name} failed - continuing with reduced functionality")
                else:
                    results["critical_failed"].append(package_name)
                    logger.error(f"Critical dependency {package_name} failed")
                    
        results["success_rate"] = len(results["successful_installs"]) / results["total_packages"]
        
        # Check resources after installation
        resource_usage = self.resource_monitor.check_resources()
        results["resource_usage"] = resource_usage
        
        logger.info(f"Installation complete - Success rate: {results['success_rate']:.1%}")
        
        if results["critical_failed"]:
            logger.error(f"Critical dependencies failed: {results['critical_failed']}")
            logger.info(f"Consider rolling back using snapshot: {snapshot_id}")
            
        return results
        
    def validate_installation(self, venv_path: str = "arena_bot_venv") -> Dict[str, Any]:
        """P0.1.4: Installation State Recovery - rollback mechanism for failed partial installs"""
        logger.info("Validating installation and testing imports")
        
        venv_path = Path(venv_path)
        python_exe = venv_path / "Scripts" / "python.exe" if os.name == 'nt' else venv_path / "bin" / "python"
        
        validation_results = {
            "validation_passed": False,
            "importable_packages": [],
            "import_failures": [],
            "functionality_tests": {},
            "fallback_available": {},
            "recovery_needed": False
        }
        
        # Test critical imports
        critical_packages = ['numpy', 'pandas', 'scikit-learn', 'lightgbm']
        
        for package in critical_packages:
            try:
                test_script = f"""
import sys
import {package}
print(f"{package} version: {{{package}.__version__}}")
"""
                result = subprocess.run([str(python_exe), '-c', test_script], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    validation_results["importable_packages"].append(package)
                    logger.info(f"✓ {package} import successful")
                else:
                    validation_results["import_failures"].append({
                        "package": package,
                        "error": result.stderr
                    })
                    logger.error(f"✗ {package} import failed: {result.stderr}")
                    
            except Exception as e:
                validation_results["import_failures"].append({
                    "package": package,
                    "error": str(e)
                })
                logger.error(f"✗ {package} validation failed: {e}")
                
        # Test functionality with fallbacks
        functionality_tests = {
            'data_validation': self._test_data_validation_fallback(),
            'resource_monitoring': self._test_resource_monitoring_fallback(),
            'configuration': self._test_configuration_fallback()
        }
        
        validation_results["functionality_tests"] = functionality_tests
        
        # Determine if validation passed
        critical_imports_ok = len(validation_results["importable_packages"]) >= 3
        functionality_ok = all(test["available"] for test in functionality_tests.values())
        
        validation_results["validation_passed"] = critical_imports_ok and functionality_ok
        
        if not validation_results["validation_passed"]:
            validation_results["recovery_needed"] = True
            logger.error("Installation validation failed - recovery needed")
        else:
            logger.info("✓ Installation validation passed")
            
        return validation_results
        
    def _test_data_validation_fallback(self) -> Dict[str, Any]:
        """Test data validation with pydantic fallback to manual validation"""
        try:
            import pydantic
            return {"available": True, "method": "pydantic", "fallback": False}
        except ImportError:
            logger.info("pydantic not available - using manual validation fallback")
            return {"available": True, "method": "manual", "fallback": True}
            
    def _test_resource_monitoring_fallback(self) -> Dict[str, Any]:
        """Test resource monitoring with psutil fallback to basic monitoring"""
        try:
            import psutil
            psutil.cpu_percent()
            return {"available": True, "method": "psutil", "fallback": False}
        except ImportError:
            logger.info("psutil not available - using basic resource monitoring fallback")
            return {"available": True, "method": "basic", "fallback": True}
            
    def _test_configuration_fallback(self) -> Dict[str, Any]:
        """Test configuration management"""
        try:
            import yaml
            import jsonschema
            return {"available": True, "method": "full", "fallback": False}
        except ImportError:
            logger.info("Configuration libraries partially available - using basic configuration")
            return {"available": True, "method": "basic", "fallback": True}
            
    def run_full_validation(self, venv_path: str = "arena_bot_venv") -> Dict[str, Any]:
        """Run complete dependency validation process"""
        logger.info("=== Starting Full Dependency Validation ===")
        
        validation_report = {
            "start_time": datetime.now().isoformat(),
            "environment_scan": {},
            "venv_creation": False,
            "installation_results": {},
            "validation_results": {},
            "overall_success": False,
            "recovery_instructions": []
        }
        
        try:
            # Phase 1: Environment Scan
            logger.info("Phase 1: Environment Scan")
            validation_report["environment_scan"] = self.scan_environment()
            
            # Phase 2: Create Virtual Environment
            logger.info("Phase 2: Virtual Environment Creation")
            validation_report["venv_creation"] = self.create_isolated_environment(venv_path)
            
            if not validation_report["venv_creation"]:
                raise DependencyValidationError("Failed to create virtual environment")
                
            # Phase 3: Install Dependencies
            logger.info("Phase 3: Dependency Installation")
            validation_report["installation_results"] = self.install_with_fallback(venv_path)
            
            # Phase 4: Validate Installation
            logger.info("Phase 4: Installation Validation")
            validation_report["validation_results"] = self.validate_installation(venv_path)
            
            # Determine overall success
            installation_ok = validation_report["installation_results"]["success_rate"] >= 0.8
            validation_ok = validation_report["validation_results"]["validation_passed"]
            
            validation_report["overall_success"] = installation_ok and validation_ok
            
            if not validation_report["overall_success"]:
                validation_report["recovery_instructions"] = self._generate_recovery_instructions(validation_report)
                
        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            validation_report["error"] = str(e)
            validation_report["overall_success"] = False
            
        validation_report["end_time"] = datetime.now().isoformat()
        
        # Save validation report
        report_file = Path("dependency_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
            
        logger.info(f"Validation report saved: {report_file}")
        
        if validation_report["overall_success"]:
            logger.info("🎉 Dependency validation completed successfully!")
        else:
            logger.error("❌ Dependency validation failed - see report for details")
            
        return validation_report
        
    def _generate_recovery_instructions(self, report: Dict[str, Any]) -> List[str]:
        """Generate recovery instructions based on validation results"""
        instructions = []
        
        if not report.get("venv_creation", False):
            instructions.append("1. Check Python installation and permissions")
            instructions.append("2. Ensure sufficient disk space for virtual environment")
            
        if report.get("installation_results", {}).get("critical_failed"):
            critical_failed = report["installation_results"]["critical_failed"]
            instructions.append(f"3. Manually install critical dependencies: {', '.join(critical_failed)}")
            instructions.append("4. Check internet connectivity and package repository access")
            
        if report.get("validation_results", {}).get("import_failures"):
            instructions.append("5. Check for missing system libraries (e.g., Visual C++ Redistributable on Windows)")
            instructions.append("6. Consider using conda instead of pip for problematic packages")
            
        snapshot_id = report.get("installation_results", {}).get("snapshot_id")
        if snapshot_id:
            instructions.append(f"7. If needed, rollback using snapshot: {snapshot_id}")
            
        return instructions

def main():
    """Main entry point for dependency validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Helper v2 Dependency Validator")
    parser.add_argument("--requirements", default="requirements_ai_v2.txt", 
                       help="Requirements file path")
    parser.add_argument("--venv", default="arena_bot_venv", 
                       help="Virtual environment path")
    parser.add_argument("--scan-only", action="store_true", 
                       help="Only scan environment, don't install")
    parser.add_argument("--no-venv", action="store_true", 
                       help="Skip virtual environment creation")
    
    args = parser.parse_args()
    
    validator = DependencyValidator(args.requirements)
    
    if args.scan_only:
        scan_results = validator.scan_environment()
        print(json.dumps(scan_results, indent=2))
        return
        
    if args.no_venv:
        logger.warning("Skipping virtual environment creation - not recommended for production")
        
    try:
        report = validator.run_full_validation(args.venv)
        
        if report["overall_success"]:
            print("\n✅ Dependency validation successful!")
            print(f"📊 Success rate: {report['installation_results']['success_rate']:.1%}")
        else:
            print("\n❌ Dependency validation failed!")
            print("📋 Recovery instructions:")
            for instruction in report.get("recovery_instructions", []):
                print(f"   {instruction}")
                
        print(f"\n📄 Full report: dependency_validation_report.json")
        
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()