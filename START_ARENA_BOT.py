#!/usr/bin/env python3
"""
DOUBLE-CLICK TO START ARENA BOT
Simply double-click this file to start the Enhanced Arena Bot!
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🎯 Starting Enhanced Arena Bot...")
    print("=" * 50)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bot_script = script_dir / "enhanced_realtime_arena_bot.py"
    
    # Change to the bot directory
    os.chdir(script_dir)
    
    # Use virtual environment python
    venv_python = script_dir / "arena_venv" / "bin" / "python"
    
    try:
        # Force check if venv exists and use it
        print(f"🔍 Checking for virtual environment at: {venv_python}")
        if venv_python.exists():
            print("✅ Using virtual environment with all dependencies")
            # Run with virtual environment
            env = os.environ.copy()
            env['PATH'] = str(script_dir / "arena_venv" / "bin") + ":" + env.get('PATH', '')
            subprocess.run([str(venv_python), str(bot_script)], env=env, check=True)
        else:
            print("⚠️ Virtual environment not found, using system python")
            print("💡 Try running START_ARENA_BOT_VENV.bat instead for full functionality")
            subprocess.run([sys.executable, str(bot_script)], check=True)
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("Press Enter to close...")
        input()

if __name__ == "__main__":
    main()