#!/usr/bin/env python3
"""
Smart OpenCV Installation for Testing

Only installs opencv-python-headless when running tests,
keeps regular opencv-python for normal development.
"""

import subprocess
import sys
import os

def check_opencv_installed():
    """Check which OpenCV version is installed"""
    try:
        result = subprocess.run([sys.executable, "-c", "import cv2; print(cv2.__file__)"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            cv2_path = result.stdout.strip()
            if 'headless' in cv2_path:
                return 'headless'
            else:
                return 'regular'
    except:
        pass
    return None

def install_opencv_headless():
    """Install opencv-python-headless for testing"""
    print("🔄 Installing opencv-python-headless for testing...")
    
    # Uninstall regular opencv if present
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"], 
                  capture_output=True)
    
    # Install headless version
    result = subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.1.78"])
    
    if result.returncode == 0:
        print("✅ opencv-python-headless installed successfully")
        return True
    else:
        print("❌ Failed to install opencv-python-headless")
        return False

def restore_opencv_regular():
    """Restore regular opencv-python after testing"""
    print("🔄 Restoring regular opencv-python...")
    
    # Uninstall headless
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python-headless"], 
                  capture_output=True)
    
    # Install regular version
    result = subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python==4.8.1.78"])
    
    if result.returncode == 0:
        print("✅ opencv-python restored successfully")
        return True
    else:
        print("❌ Failed to restore opencv-python")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--for-testing", action="store_true", help="Install headless for testing")
    parser.add_argument("--restore", action="store_true", help="Restore regular opencv")
    parser.add_argument("--check", action="store_true", help="Check current installation")
    
    args = parser.parse_args()
    
    if args.check:
        current = check_opencv_installed()
        if current:
            print(f"📊 Current OpenCV: {current}")
        else:
            print("❌ OpenCV not installed")
    
    elif args.for_testing:
        if install_opencv_headless():
            print("🧪 Ready for headless testing!")
            print("💡 Remember to run --restore when done")
        else:
            sys.exit(1)
    
    elif args.restore:
        if restore_opencv_regular():
            print("🖥️ Ready for GUI development!")
        else:
            sys.exit(1)
    
    else:
        print("Usage:")
        print("  python install_test_deps.py --check")
        print("  python install_test_deps.py --for-testing")
        print("  python install_test_deps.py --restore")

if __name__ == "__main__":
    main()