#!/usr/bin/env python3
"""
Basic validation script for Phase 0 components
Tests core functionality without external dependencies
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Phase 0 modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from arena_bot.ai_v2.data_models import (
            CardInfo, CardOption, EvaluationScores, AIDecision, BaseDataModel
        )
        print("✅ data_models imported successfully")
    except Exception as e:
        print(f"❌ data_models import failed: {e}")
        return False
    
    try:
        from arena_bot.config.config_manager import ConfigurationManager
        print("✅ config_manager imported successfully")
    except Exception as e:
        print(f"❌ config_manager import failed: {e}")
        return False
    
    try:
        from arena_bot.ai_v2.monitoring import PerformanceMonitor
        print("✅ monitoring imported successfully")
    except Exception as e:
        print(f"❌ monitoring import failed: {e}")
        return False
    
    try:
        from scripts.validate_dependencies import DependencyValidator
        print("✅ dependency_validation imported successfully")
    except Exception as e:
        print(f"❌ dependency_validation import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of each component"""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test data models
        from arena_bot.ai_v2.data_models import CardInfo, AIDecision
        
        # Test CardInfo
        card = CardInfo(
            name="Lightning Bolt",
            cost=1,
            card_class="Red",
            rarity="Common",
            card_type="Instant"
        )
        
        # Test serialization
        card_dict = card.to_dict()
        card_restored = CardInfo.from_dict(card_dict)
        
        assert card.name == card_restored.name
        assert card.cost == card_restored.cost
        print("✅ Data models working correctly")
        
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test config manager
        from arena_bot.config.config_manager import ConfigurationManager
        
        config = ConfigurationManager()
        
        # Test basic operations
        config.update_config({"test": {"key": "test_value"}})
        value = config.get_config("test.key", "default")
        assert value == "test_value"
        
        print("✅ Config manager working correctly")
        
    except Exception as e:
        print(f"❌ Config manager test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test monitoring
        from arena_bot.ai_v2.monitoring import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Test basic monitoring
        with monitor.measure_time("test_operation"):
            pass
        
        metrics = monitor.get_metrics_summary()
        assert "test_operation" in metrics
        
        print("✅ Monitoring working correctly")
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test dependency validation
        from scripts.validate_dependencies import DependencyValidator
        
        validator = DependencyValidator()
        
        # Test basic validation
        result = validator.validate_runtime_dependencies()
        print(f"✅ Dependency validation working correctly (critical deps: {result.has_critical_issues})")
        
    except Exception as e:
        print(f"❌ Dependency validation test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_integration():
    """Test integration between components"""
    print("🔗 Testing component integration...")
    
    try:
        from arena_bot.ai_v2.data_models import CardInfo, AIDecision  
        from arena_bot.config.config_manager import ConfigurationManager
        from arena_bot.ai_v2.monitoring import PerformanceMonitor
        
        # Initialize components
        config = ConfigurationManager()
        monitor = PerformanceMonitor()
        
        # Test configuration for AI components
        config.update_config({"ai": {"max_tokens": 4000, "temperature": 0.7}})
        
        # Test monitoring with configuration
        with monitor.measure_time("ai_request_simulation"):
            max_tokens = config.get_config("ai.max_tokens", 2000)
            temperature = config.get_config("ai.temperature", 0.5)
            
            # Simulate card analysis processing
            card = CardInfo(
                name="Integration Test Card",
                cost=max_tokens // 1000,  # Use config value in some way
                card_type="Spell"
            )
        
        metrics = monitor.get_metrics_summary()
        assert "ai_request_simulation" in metrics
        
        print("✅ Component integration working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("🚀 Starting Phase 0 Basic Validation")
    print("=" * 50)
    
    success = True
    
    success &= test_imports()
    print()
    
    success &= test_basic_functionality()
    print()
    
    success &= test_integration()
    print()
    
    if success:
        print("🎉 All Phase 0 validation tests passed!")
        print("Phase 0 implementation is ready for integration")
        return 0
    else:
        print("❌ Some validation tests failed")
        print("Phase 0 implementation needs fixes before integration")
        return 1

if __name__ == "__main__":
    sys.exit(main())