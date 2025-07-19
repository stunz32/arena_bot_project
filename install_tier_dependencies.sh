#!/bin/bash

echo "===================================="
echo "Installing Tier Integration Dependencies"
echo "===================================="
echo ""
echo "This will install the required packages for HearthArena tier integration:"
echo "- beautifulsoup4 (HTML parsing)"
echo "- requests (HTTP requests)"
echo "- rapidfuzz (fuzzy string matching)"
echo "- lxml (faster XML/HTML parsing)"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found! Please install Python 3.7+."
        exit 1
    else
        PYTHON_CMD="python"
        PIP_CMD="pip"
    fi
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

echo "Installing dependencies..."
echo ""

# Install the required packages
$PIP_CMD install -r requirements_tier_integration.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies!"
    echo ""
    echo "Try using:"
    echo "  $PIP_CMD install --user beautifulsoup4 requests rapidfuzz lxml"
    echo ""
    exit 1
fi

echo ""
echo "===================================="
echo "Dependencies installed successfully!"
echo "===================================="
echo ""
echo "You can now run:"
echo "  $PYTHON_CMD test_tier_integration.py"
echo "  $PYTHON_CMD enhanced_arena_bot_with_tiers.py"
echo ""