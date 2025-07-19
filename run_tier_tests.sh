#!/bin/bash

echo "===================================="
echo "HearthArena Tier Integration Tests"
echo "===================================="
echo ""
echo "This will test the new tier integration features."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found! Please install Python 3.7+."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Testing tier integration..."
echo ""

# Run the tier integration test
$PYTHON_CMD test_tier_integration.py

echo ""
echo "===================================="
echo "Tests completed!"
echo "===================================="
echo ""
echo "If tests passed, you can now run:"
echo "  $PYTHON_CMD enhanced_arena_bot_with_tiers.py"
echo ""