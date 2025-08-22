#!/bin/bash

# Setup Test Environment for Arena Bot
# Implements your friend's recommendations for fixing Qt/OpenCV conflicts

set -e

echo "🔧 Setting up Arena Bot test environment..."

# Install system dependencies for Qt6 and X11 (for WSL2/headless)
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    xvfb \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libxcb1 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-icccm4 \
    libxcb-randr0 \
    libxcb-util1 \
    libxcb-cursor0 \
    libopengl0 \
    libegl1 \
    mesa-utils

echo "🐍 Setting up Python testing environment..."

# Remove conflicting OpenCV if present
if pip show opencv-python >/dev/null 2>&1; then
    echo "⚠️ Removing opencv-python to prevent Qt conflicts..."
    pip uninstall -y opencv-python
fi

# Install testing dependencies
pip install -r requirements-test.txt

# Create test environment variables script
cat > test_env.sh << 'EOF'
#!/bin/bash
# Test Environment Variables
# Source this before running tests: source test_env.sh

# Prevent Qt plugin conflicts
export QT_PLUGIN_PATH=
export QT_DEBUG_PLUGINS=0

# Use software rendering for headless
export LIBGL_ALWAYS_SOFTWARE=1

# Qt scaling for consistent test results
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_ENABLE_HIGHDPI_SCALING=1
export QT_SCALE_FACTOR=1.0

# Test profile for fast startup
export TEST_PROFILE=1

# Xvfb display
export DISPLAY=:99

echo "✅ Test environment variables loaded"
EOF

chmod +x test_env.sh

# Create test runner script
cat > run_tests.sh << 'EOF'
#!/bin/bash
# Test Runner with Xvfb
# Implements headless testing with virtual display

source ./test_env.sh

echo "🧪 Starting Arena Bot tests in headless mode..."

# Start Xvfb in background if not running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "📺 Starting virtual display..."
    Xvfb :99 -screen 0 1920x1200x24 -ac &
    XVFB_PID=$!
    sleep 2
fi

# Run tests with proper environment
echo "🎮 Running comprehensive tests..."
python3 test_comprehensive_bot.py --auto-fix

echo "🎯 Running user interaction tests..."
python3 test_user_interactions_lightweight.py --all

# Cleanup
if [ ! -z "$XVFB_PID" ]; then
    kill $XVFB_PID 2>/dev/null || true
fi

echo "✅ Test run complete"
EOF

chmod +x run_tests.sh

echo "✅ Test environment setup complete!"
echo ""
echo "📋 Usage:"
echo "  source test_env.sh      # Load environment variables"
echo "  ./run_tests.sh          # Run all tests in headless mode"
echo "  xvfb-run python3 test_comprehensive_bot.py  # Run individual tests"
echo ""
echo "🔍 Next steps:"
echo "  1. Run: source test_env.sh"
echo "  2. Run: ./run_tests.sh"
echo "  3. Check artifacts/ for test reports"