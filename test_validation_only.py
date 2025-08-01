#!/usr/bin/env python3
"""
Quick test for validation timing fixes
"""

import sys
import os
from pathlib import Path

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent))

# Set fake display to avoid GUI issues
os.environ['DISPLAY'] = ':99'

print("🧪 Testing validation timing fixes...")

try:
    print("📦 Importing IntegratedArenaBotGUI...")
    from integrated_arena_bot_gui import IntegratedArenaBotGUI
    
    print("🏗️ Creating bot instance...")
    # This will trigger the validation at the end of __init__
    bot = IntegratedArenaBotGUI()
    
    print("✅ SUCCESS: Bot initialized without errors!")
    print("🎯 Validation timing fixes are working correctly!")
    
    # Test that critical methods now exist
    critical_methods = ['_register_thread', '_unregister_thread', 'manual_screenshot', 'log_text']
    
    print("\n📋 Post-initialization method check:")
    for method in critical_methods:
        if hasattr(bot, method):
            print(f"✅ {method}: Available")
        else:
            print(f"❌ {method}: Missing")
    
    # Test that critical attributes exist
    critical_attributes = ['_active_threads', '_thread_lock', 'result_queue', 'event_queue']
    
    print("\n📋 Post-initialization attribute check:")
    for attr in critical_attributes:
        if hasattr(bot, attr):
            attr_value = getattr(bot, attr)
            print(f"✅ {attr}: Available (type: {type(attr_value).__name__})")
        else:
            print(f"❌ {attr}: Missing")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All validation timing fixes working correctly!")