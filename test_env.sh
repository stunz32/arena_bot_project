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

# Use offscreen Qt platform to avoid XCB issues
export QT_QPA_PLATFORM=offscreen

echo "✅ Test environment variables loaded"