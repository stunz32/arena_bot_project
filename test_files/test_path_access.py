#!/usr/bin/env python3
"""
Test path access to Hearthstone logs.
"""

from pathlib import Path
import os

def test_path_access():
    print("🔍 TESTING HEARTHSTONE LOG PATH ACCESS")
    print("=" * 50)
    
    # Test different path formats
    paths_to_test = [
        "/mnt/m/Hearthstone/Logs",
        "M:\\Hearthstone\\Logs",
        "/mnt/m/Hearthstone\\Logs",  # Mixed separators
    ]
    
    for path_str in paths_to_test:
        print(f"\n📂 Testing path: {path_str}")
        
        try:
            path = Path(path_str)
            print(f"   Path object created: {path}")
            
            # Test existence
            exists = path.exists()
            print(f"   Exists: {exists}")
            
            if exists:
                # Test if it's a directory
                is_dir = path.is_dir()
                print(f"   Is directory: {is_dir}")
                
                if is_dir:
                    # Test listing contents
                    try:
                        items = list(path.iterdir())
                        print(f"   Items found: {len(items)}")
                        
                        # Show first few items
                        for i, item in enumerate(items[:3]):
                            print(f"      {i+1}. {item.name}")
                            
                    except Exception as e:
                        print(f"   ❌ Cannot list directory: {e}")
                        
        except Exception as e:
            print(f"   ❌ Path error: {e}")
    
    # Test manual bash access
    print(f"\n🔧 Manual bash test:")
    try:
        import subprocess
        result = subprocess.run(['ls', '-la', '/mnt/m/Hearthstone/Logs'], 
                              capture_output=True, text=True, timeout=10)
        print(f"   Exit code: {result.returncode}")
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print(f"   Found {len(lines)} lines")
            for line in lines[:5]:  # First 5 lines
                print(f"      {line}")
        if result.stderr:
            print(f"   Stderr: {result.stderr}")
            
    except Exception as e:
        print(f"   ❌ Bash test error: {e}")

if __name__ == "__main__":
    test_path_access()