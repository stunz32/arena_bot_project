@echo off
title Arena Bot - Native Windows Version
color 0A
echo.
echo ========================================
echo   ARENA BOT - NATIVE WINDOWS VERSION
echo ========================================
echo.
echo Starting Arena Bot with native Windows Python...
echo No WSL, no X server, no complexity!
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found!
    echo 💡 Please install Python from https://python.org
    echo 💡 Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "enhanced_realtime_arena_bot.py" (
    echo ❌ Bot files not found in current directory!
    echo 💡 Make sure you're running this from D:\cursor bots\arena_bot_project\
    echo.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Checking Python dependencies...
pip install -q -r requirements_windows.txt

REM Run the bot
echo.
echo 🚀 Launching Arena Bot...
echo ✅ Windows native Python
echo ✅ No WSL required
echo ✅ GUI should open automatically
echo.
python enhanced_realtime_arena_bot.py

echo.
echo ========================================
echo Arena Bot finished.
pause