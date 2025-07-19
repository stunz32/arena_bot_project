@echo off
echo ====================================
echo HearthArena Tier Integration Tests
echo ====================================
echo.
echo This will test the new tier integration features.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.7+ and add it to PATH.
    pause
    exit /b 1
)

echo Testing tier integration...
echo.

REM Run the tier integration test
python test_tier_integration.py

echo.
echo ====================================
echo Tests completed!
echo ====================================
echo.
echo If tests passed, you can now run:
echo   python enhanced_arena_bot_with_tiers.py
echo.
pause