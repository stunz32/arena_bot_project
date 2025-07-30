#!/usr/bin/env python3
"""
Test suite for Dependency Validation System
Tests the comprehensive dependency management with hardening features
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Import the dependency validation components
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from validate_dependencies import (
    DependencyValidationError, InstallationStateManager, 
    ResourceMonitor, DependencyValidator
)


class TestInstallationStateManager:
    """Test installation state management and rollback"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.state_dir = Path(self.temp_dir) / "test_states"
        self.state_manager = InstallationStateManager(str(self.state_dir))
        
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_state_manager_initialization(self):
        """Test state manager initialization"""
        assert self.state_dir.exists()
        assert self.state_manager.current_state_file.parent == self.state_dir
        assert self.state_manager.backup_state_file.parent == self.state_dir
        
    def test_create_snapshot(self):
        """Test environment snapshot creation"""
        with patch('validate_dependencies.pkg_resources') as mock_pkg:
            # Mock installed packages
            mock_pkg.working_set = [
                MagicMock(key='numpy', version='1.26.4'),
                MagicMock(key='pandas', version='2.2.2')
            ]
            
            snapshot_id = self.state_manager.create_snapshot()
            
            assert isinstance(snapshot_id, str)
            assert len(snapshot_id) > 0
            
            # Snapshot file should be created
            snapshot_file = self.state_dir / f"snapshot_{snapshot_id}.json"
            assert snapshot_file.exists()
            
            # Verify snapshot contents
            with open(snapshot_file, 'r') as f:
                snapshot_data = json.load(f)
                
            assert "snapshot_id" in snapshot_data
            assert "timestamp" in snapshot_data
            assert "python_version" in snapshot_data
            assert "installed_packages" in snapshot_data
            assert "numpy" in snapshot_data["installed_packages"]
            
    def test_rollback_to_snapshot(self):
        """Test rollback to previous snapshot"""
        # Create a test snapshot
        snapshot_id = self.state_manager.create_snapshot()
        
        # Test rollback
        result = self.state_manager.rollback_to_snapshot(snapshot_id)
        assert result is True
        
        # Test rollback to non-existent snapshot
        result = self.state_manager.rollback_to_snapshot("non_existent_id")
        assert result is False


class TestResourceMonitor:
    """Test resource monitoring during installation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.resource_monitor = ResourceMonitor()
        
    @patch('validate_dependencies.psutil')
    def test_resource_monitoring_with_psutil(self, mock_psutil):
        """Test resource monitoring when psutil is available"""
        # Mock psutil functions
        mock_memory = MagicMock()
        mock_memory.used = 1024 * 1024 * 1024  # 1GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = MagicMock()
        mock_disk.used = 10 * 1024 * 1024 * 1024  # 10GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_psutil.cpu_percent.return_value = 25.5
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Check resources
        resources = self.resource_monitor.check_resources()
        
        assert "memory_used_mb" in resources
        assert "disk_used_gb" in resources
        assert "cpu_percent" in resources
        assert resources["cpu_percent"] == 25.5
        
    def test_resource_monitoring_without_psutil(self):
        """Test resource monitoring fallback when psutil unavailable"""
        # psutil not available (default behavior)
        self.resource_monitor.start_monitoring()
        
        resources = self.resource_monitor.check_resources()
        assert resources == {"status": "monitoring_unavailable"}


class TestDependencyValidator:
    """Test main dependency validation functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.requirements_file = Path(self.temp_dir) / "test_requirements.txt"
        
        # Create test requirements file
        test_requirements = """
# Test requirements file
numpy==1.26.4
pandas==2.2.2
psutil==5.9.8  # Optional dependency
pytest==8.1.1  # Development dependency
"""
        with open(self.requirements_file, 'w') as f:
            f.write(test_requirements)
            
        self.validator = DependencyValidator(str(self.requirements_file))
        
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_validator_initialization(self):
        """Test dependency validator initialization"""
        assert self.validator.requirements_file == self.requirements_file
        assert isinstance(self.validator.state_manager, InstallationStateManager)
        assert isinstance(self.validator.resource_monitor, ResourceMonitor)
        assert len(self.validator.optional_deps) > 0
        
    @patch('validate_dependencies.pkg_resources')
    @patch('validate_dependencies.psutil')
    def test_environment_scan(self, mock_psutil, mock_pkg):
        """Test P0.1.1: Pre-installation Environment Scan"""
        # Mock system resources
        mock_disk = MagicMock()
        mock_disk.free = 50 * 1024**3  # 50GB free
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_memory = MagicMock()
        mock_memory.available = 8 * 1024**3  # 8GB available
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock installed packages with conflicts
        mock_pkg.working_set = [
            MagicMock(key='numpy', version='2.0.0'),  # Conflict: requires 1.26.4
            MagicMock(key='pandas', version='2.2.2'),  # No conflict
        ]
        mock_pkg.get_distribution.side_effect = lambda name: MagicMock(version='2.0.0') if name == 'numpy' else MagicMock(version='2.2.2')
        
        scan_results = self.validator.scan_environment()
        
        assert "python_version" in scan_results
        assert "pip_version" in scan_results
        assert "potential_conflicts" in scan_results
        assert "system_packages" in scan_results
        assert "disk_space_gb" in scan_results
        assert "memory_available_gb" in scan_results
        
        # Should detect numpy version conflict
        conflicts = scan_results["potential_conflicts"]
        numpy_conflict = next((c for c in conflicts if c["package"] == "numpy"), None)
        assert numpy_conflict is not None
        assert numpy_conflict["required"] == "1.26.4"
        assert numpy_conflict["current"] == "2.0.0"
        assert numpy_conflict["severity"] == "high"
        
    @patch('validate_dependencies.venv')
    @patch('validate_dependencies.subprocess')
    def test_create_isolated_environment(self, mock_subprocess, mock_venv):
        """Test P0.1.2: Virtual Environment Isolation"""
        venv_path = Path(self.temp_dir) / "test_venv"
        
        # Mock successful venv creation
        mock_venv.create.return_value = None
        
        # Mock successful pip test
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.run.return_value = mock_result
        
        # Create mock python executable
        if os.name == 'nt':
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        python_exe.touch()
        
        result = self.validator.create_isolated_environment(str(venv_path))
        
        assert result is True
        mock_venv.create.assert_called_once_with(venv_path, with_pip=True, clear=True)
        
    @patch('validate_dependencies.subprocess')
    def test_install_with_fallback(self, mock_subprocess):
        """Test P0.1.3: Graceful Dependency Fallback"""
        venv_path = Path(self.temp_dir) / "test_venv"
        
        # Create mock venv structure
        if os.name == 'nt':
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        python_exe.touch()
        
        # Mock installation results (some succeed, some fail)
        def mock_run(*args, **kwargs):
            cmd = args[0]
            if 'numpy' in ' '.join(cmd):
                # numpy installation succeeds
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                return result
            elif 'pytest' in ' '.join(cmd):
                # pytest (optional) fails
                result = MagicMock()
                result.returncode = 1
                result.stderr = "Installation failed"
                return result
            else:
                # Other packages succeed
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                return result
                
        mock_subprocess.run.side_effect = mock_run
        
        with patch.object(self.validator.resource_monitor, 'start_monitoring'):
            with patch.object(self.validator.resource_monitor, 'check_resources', return_value={}):
                results = self.validator.install_with_fallback(str(venv_path))
        
        assert "successful_installs" in results
        assert "failed_installs" in results
        assert "optional_failed" in results
        assert "critical_failed" in results
        assert "success_rate" in results
        
        # numpy should be in successful installs
        assert "numpy" in results["successful_installs"]
        
        # pytest should be in optional failed (since it's in optional_deps)
        assert "pytest" in results["optional_failed"]
        
    @patch('validate_dependencies.subprocess')
    def test_validate_installation(self, mock_subprocess):
        """Test P0.1.4: Installation State Recovery"""
        venv_path = Path(self.temp_dir) / "test_venv"
        
        # Create mock venv structure
        if os.name == 'nt':
            python_exe = venv_path / "Scripts" / "python.exe"
        else:
            python_exe = venv_path / "bin" / "python"
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        python_exe.touch()
        
        # Mock successful import tests
        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result
            
        mock_subprocess.run.side_effect = mock_run
        
        validation_results = self.validator.validate_installation(str(venv_path))
        
        assert "validation_passed" in validation_results
        assert "importable_packages" in validation_results
        assert "import_failures" in validation_results
        assert "functionality_tests" in validation_results
        
        # Should test data validation, resource monitoring, and configuration
        assert "data_validation" in validation_results["functionality_tests"]
        assert "resource_monitoring" in validation_results["functionality_tests"]
        assert "configuration" in validation_results["functionality_tests"]
        
    def test_fallback_functionality_tests(self):
        """Test fallback functionality detection"""
        # Test data validation fallback
        data_val_result = self.validator._test_data_validation_fallback()
        assert "available" in data_val_result
        assert "method" in data_val_result
        assert "fallback" in data_val_result
        
        # Test resource monitoring fallback
        resource_result = self.validator._test_resource_monitoring_fallback()
        assert "available" in resource_result
        assert "method" in resource_result
        
        # Test configuration fallback
        config_result = self.validator._test_configuration_fallback()
        assert "available" in config_result
        assert "method" in config_result
        
    @patch('validate_dependencies.subprocess')
    @patch('validate_dependencies.venv')
    def test_full_validation_workflow(self, mock_venv, mock_subprocess):
        """Test complete validation workflow"""
        venv_path = "test_full_venv"
        
        # Mock all subprocess calls to succeed
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.run.return_value = mock_result
        
        # Mock venv creation
        mock_venv.create.return_value = None
        
        # Create mock python executable in expected location
        if os.name == 'nt':
            python_exe = Path(venv_path) / "Scripts" / "python.exe"
        else:
            python_exe = Path(venv_path) / "bin" / "python"
        python_exe.parent.mkdir(parents=True, exist_ok=True)
        python_exe.touch()
        
        try:
            with patch.object(self.validator.resource_monitor, 'start_monitoring'):
                with patch.object(self.validator.resource_monitor, 'check_resources', return_value={}):
                    with patch('validate_dependencies.psutil', None):  # Test without psutil
                        report = self.validator.run_full_validation(venv_path)
            
            assert "start_time" in report
            assert "end_time" in report
            assert "environment_scan" in report
            assert "venv_creation" in report
            assert "installation_results" in report
            assert "validation_results" in report
            assert "overall_success" in report
            
            # Should have created validation report file
            report_file = Path("dependency_validation_report.json")
            if report_file.exists():
                report_file.unlink()  # Cleanup
                
        finally:
            # Cleanup created venv directory
            if Path(venv_path).exists():
                shutil.rmtree(venv_path, ignore_errors=True)
                
    def test_recovery_instructions_generation(self):
        """Test recovery instructions generation"""
        # Mock failed validation report
        failed_report = {
            "venv_creation": False,
            "installation_results": {
                "critical_failed": ["numpy", "pandas"],
                "snapshot_id": "test_snapshot_123"
            },
            "validation_results": {
                "import_failures": [
                    {"package": "lightgbm", "error": "Missing system libraries"}
                ]
            }
        }
        
        instructions = self.validator._generate_recovery_instructions(failed_report)
        
        assert len(instructions) > 0
        
        # Should include specific recovery steps
        instruction_text = ' '.join(instructions)
        assert "Python installation" in instruction_text
        assert "numpy, pandas" in instruction_text
        assert "test_snapshot_123" in instruction_text
        assert "system libraries" in instruction_text
        
    def test_validation_error_handling(self):
        """Test proper error handling"""
        # Test with non-existent requirements file
        bad_validator = DependencyValidator("non_existent_requirements.txt")
        
        with pytest.raises(DependencyValidationError):
            bad_validator.install_with_fallback("test_venv")
            
    def test_optional_dependency_classification(self):
        """Test that optional dependencies are correctly classified"""
        assert "psutil" in self.validator.optional_deps
        assert "pytest" in self.validator.optional_deps
        assert "numpy" not in self.validator.optional_deps  # Critical dependency


class TestCommandLineInterface:
    """Test command line interface functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.requirements_file = Path(self.temp_dir) / "cli_test_requirements.txt"
        
        # Create minimal requirements file
        with open(self.requirements_file, 'w') as f:
            f.write("numpy==1.26.4\n")
            
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('validate_dependencies.DependencyValidator')
    def test_scan_only_mode(self, mock_validator_class):
        """Test --scan-only command line option"""
        mock_validator = MagicMock()
        mock_scan_results = {
            "python_version": "3.12.3",
            "potential_conflicts": [],
            "disk_space_gb": 50.0
        }
        mock_validator.scan_environment.return_value = mock_scan_results
        mock_validator_class.return_value = mock_validator
        
        # Import and test main function
        from validate_dependencies import main
        
        with patch('sys.argv', ['validate_dependencies.py', '--scan-only', '--requirements', str(self.requirements_file)]):
            with patch('builtins.print') as mock_print:
                main()
                
        # Should have called scan_environment and printed results
        mock_validator.scan_environment.assert_called_once()
        mock_print.assert_called()
        
    @patch('validate_dependencies.DependencyValidator')
    def test_full_validation_mode(self, mock_validator_class):
        """Test full validation mode"""
        mock_validator = MagicMock()
        mock_report = {
            "overall_success": True,
            "installation_results": {"success_rate": 0.9},
            "recovery_instructions": []
        }
        mock_validator.run_full_validation.return_value = mock_report
        mock_validator_class.return_value = mock_validator
        
        from validate_dependencies import main
        
        with patch('sys.argv', ['validate_dependencies.py', '--requirements', str(self.requirements_file)]):
            with patch('builtins.print') as mock_print:
                main()
                
        # Should have called run_full_validation
        mock_validator.run_full_validation.assert_called_once()
        mock_print.assert_called()


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup after integration tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_partial_installation_recovery(self):
        """Test recovery from partial installation failure"""
        requirements_file = Path(self.temp_dir) / "partial_requirements.txt"
        
        # Create requirements with mix of available and unavailable packages
        with open(requirements_file, 'w') as f:
            f.write("json==2.0.9\n")  # Built-in, should "fail" to install
            f.write("os==1.0.0\n")   # Built-in, should "fail" to install
            
        validator = DependencyValidator(str(requirements_file))
        
        # Mock state manager
        with patch.object(validator.state_manager, 'create_snapshot', return_value="test_snapshot"):
            with patch.object(validator.resource_monitor, 'start_monitoring'):
                with patch.object(validator.resource_monitor, 'check_resources', return_value={}):
                    # This would normally fail, but we test the error handling
                    with pytest.raises(DependencyValidationError):
                        validator.install_with_fallback("non_existent_venv")
                        
    def test_environment_scan_with_conflicts(self):
        """Test environment scan detecting realistic conflicts"""
        requirements_file = Path(self.temp_dir) / "conflict_requirements.txt"
        
        with open(requirements_file, 'w') as f:
            f.write("numpy==1.20.0\n")  # Older version to create conflict
            
        validator = DependencyValidator(str(requirements_file))
        
        # Mock current numpy installation with newer version
        with patch('validate_dependencies.pkg_resources') as mock_pkg:
            mock_pkg.working_set = [MagicMock(key='numpy', version='1.26.4')]
            mock_pkg.get_distribution.return_value = MagicMock(version='1.26.4')
            
            scan_results = validator.scan_environment()
            
            # Should detect version conflict
            conflicts = scan_results["potential_conflicts"]
            assert len(conflicts) > 0
            
            numpy_conflict = conflicts[0]
            assert numpy_conflict["package"] == "numpy"
            assert numpy_conflict["required"] == "1.20.0"
            assert numpy_conflict["current"] == "1.26.4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])