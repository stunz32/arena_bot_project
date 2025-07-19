@echo off
echo ====================================
echo Installing Tier Integration Dependencies
echo ====================================
echo.
echo This will install the required packages for HearthArena tier integration:
echo - beautifulsoup4 (HTML parsing)
echo - requests (HTTP requests)  
echo - rapidfuzz (fuzzy string matching)
echo - lxml (faster XML/HTML parsing)
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.7+ and add it to PATH.
    pause
    exit /b 1
)

echo Installing dependencies...
echo.

REM Install the required packages
pip install -r requirements_tier_integration.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo.
    echo Try running as administrator or using:
    echo   pip install --user beautifulsoup4 requests rapidfuzz lxml
    echo.
    pause
    exit /b 1
)

echo.
echo ====================================
echo Dependencies installed successfully!
echo ====================================
echo.
echo You can now run:
echo   python test_tier_integration.py
echo   python enhanced_arena_bot_with_tiers.py
echo.
pause