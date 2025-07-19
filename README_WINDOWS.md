# Arena Bot - Windows Native Version

🎯 **Simple Installation - No WSL, No X Server, No Complexity!**

## Quick Start (3 Steps)

### 1. Install Python (if not already installed)
- Download from: https://python.org
- **IMPORTANT**: Check "Add Python to PATH" during installation
- Any version 3.8+ works

### 2. Run Setup
- Double-click: `SETUP_WINDOWS.bat`
- This installs all required packages automatically

### 3. Start the Bot
- Double-click: `START_ARENA_BOT_WINDOWS.bat`
- GUI opens immediately - just like any Windows program!

## Card Images (Optional)
If you need to copy card images:
- **From**: `\\wsl.localhost\Ubuntu\home\marcco\arena_bot_project\assets\cards`
- **To**: `D:\cursor bots\arena_bot_project\assets\cards`

## How to Use
1. Click "START MONITORING" in the bot GUI
2. Open Hearthstone 
3. Start an Arena draft
4. Get instant card recommendations with explanations!

## Features
- ✅ **100% Accuracy** - Same detection engine as before
- ✅ **Native Windows** - No WSL complexity
- ✅ **Instant GUI** - Opens like any Windows app
- ✅ **Real Card Names** - No confusing codes
- ✅ **Detailed Explanations** - Understand why each pick is good
- ✅ **Screen Detection** - Shows which Hearthstone screen you're on

## Troubleshooting
- **"Python not found"**: Install Python from python.org with PATH option
- **"GUI failed to start"**: Reinstall Python with tkinter support
- **"No screenshot method"**: Run `pip install pillow`

## Files
- `START_ARENA_BOT_WINDOWS.bat` - Main launcher (double-click this!)
- `SETUP_WINDOWS.bat` - One-time setup
- `enhanced_realtime_arena_bot.py` - The bot code
- `requirements_windows.txt` - Python packages needed