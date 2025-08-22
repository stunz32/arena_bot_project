#!/bin/bash
# Install Dependencies in Virtual Environment
# Fixes the externally-managed-environment issue

set -e
echo "🔧 Installing dependencies in virtual environment..."

# Ensure we're in the right directory
cd "/mnt/d/cursor bots/arena_bot_project"

# Check if virtual environment exists
if [ ! -d "test_env" ]; then
    echo "❌ Virtual environment not found. Creating new one..."
    python3 -m venv test_env
fi

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source test_env/bin/activate

# Verify we're in virtual environment
echo "📍 Current Python: $(which python)"
echo "📍 Current pip: $(which pip)"

# Upgrade pip first
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies one by one with progress
echo "📦 Installing numpy..."
pip install numpy

echo "📦 Installing OpenCV (headless)..."
pip install opencv-python-headless==4.8.1.78

echo "📦 Installing PyQt6..."
pip install PyQt6

echo "📦 Installing pytest..."
pip install pytest pytest-qt

echo "📦 Installing system utilities..."
pip install psutil Pillow

# Verify installations
echo "🔍 Verifying installations..."
python -c "import numpy; print('✅ numpy:', numpy.__version__)"
python -c "import cv2; print('✅ opencv:', cv2.__version__)"
python -c "import PyQt6; print('✅ PyQt6: installed')"
python -c "import pytest; print('✅ pytest:', pytest.__version__)"
python -c "import psutil; print('✅ psutil:', psutil.__version__)"

echo "🎉 All dependencies installed successfully!"
echo ""
echo "💡 To use this environment:"
echo "   source test_env/bin/activate"
echo "   python your_script.py"