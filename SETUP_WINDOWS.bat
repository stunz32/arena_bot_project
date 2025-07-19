@echo off
title Arena Bot - Windows Setup
color 0B
echo.
echo ========================================
echo   ARENA BOT - WINDOWS SETUP HELPER
echo ========================================
echo.
echo This script will help you set up the Arena Bot on Windows.
echo.

REM Check if Python is installed
echo 1. Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found!
    echo.
    echo 💡 Please install Python first:
    echo    1. Go to https://python.org
    echo    2. Download Python 3.8 or newer
    echo    3. Run installer and CHECK "Add Python to PATH"
    echo    4. Run this setup script again
    echo.
    pause
    exit /b 1
) else (
    python --version
    echo ✅ Python found!
)

echo.
echo 2. Installing Python packages...
pip install -r requirements_windows.txt
if %errorlevel% neq 0 (
    echo ❌ Package installation failed!
    echo 💡 Try running as administrator or check your internet connection
    pause
    exit /b 1
)
echo ✅ Packages installed successfully!

echo.
echo 3. Testing screenshot capability...
python -c "from PIL import ImageGrab; print('✅ PIL ImageGrab working'); import tkinter; print('✅ tkinter working')"
if %errorlevel% neq 0 (
    echo ❌ Some components may not work properly
    echo 💡 Try reinstalling Python with tkinter support
) else (
    echo ✅ All components working!
)

echo.
echo ========================================
echo ✅ SETUP COMPLETE!
echo ========================================
echo.
echo You can now run: START_ARENA_BOT_WINDOWS.bat
echo.
echo 📝 To copy card images:
echo Copy from: \\wsl.localhost\Ubuntu\home\marcco\arena_bot_project\assets\cards
echo Copy to:   D:\cursor bots\arena_bot_project\assets\cards
echo.
pause