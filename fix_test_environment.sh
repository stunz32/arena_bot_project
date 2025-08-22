#!/bin/bash
# Fix Test Environment Dependencies
# Addresses the missing dependencies identified in the test run

set -e
echo "🔧 Fixing test environment dependencies..."

# Activate test environment
source test_env/bin/activate

echo "📦 Installing core dependencies..."
pip install --quiet numpy
pip install --quiet opencv-python-headless==4.8.1.78
pip install --quiet PyQt6
pip install --quiet pytest pytest-qt
pip install --quiet psutil
pip install --quiet Pillow

echo "🔍 Verifying installations..."
python -c "import numpy; print('✅ numpy:', numpy.__version__)"
python -c "import cv2; print('✅ opencv:', cv2.__version__)"
python -c "import PyQt6; print('✅ PyQt6: installed')"
python -c "import pytest; print('✅ pytest:', pytest.__version__)"
python -c "import psutil; print('✅ psutil:', psutil.__version__)"

echo "✅ Test environment dependencies fixed!"