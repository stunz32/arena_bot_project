# Suggested Development Commands

## Running the Application

### Linux/WSL Commands
```bash
# Main entry point
python3 main.py

# Enhanced real-time bot
python3 enhanced_realtime_arena_bot.py

# Using shell scripts
./run_arena_bot.sh
./run_tier_tests.sh
```

### Windows Commands
```cmd
# Setup (one-time)
SETUP_WINDOWS.bat

# Start the bot GUI
START_ARENA_BOT_WINDOWS.bat

# Run tests
run_test.bat
run_tier_tests.bat

# Alternative launchers
run_enhanced_arena_bot.bat
run_py_3.11_FULLbot.bat
```

## Testing Commands

### Core Testing
```bash
# Main test suite
python3 -m pytest tests/

# Detection accuracy test
python3 test_detection_accuracy.py

# Specific test categories
python3 test_core_components.py
python3 test_bulletproof_fixes.py
python3 test_tier_integration.py
```

### Performance & Validation Tests
```bash
# Performance benchmarking
python3 test_performance_bottlenecks.py
python3 test_ultimate_performance.py

# Memory and thread safety
python3 test_memory_leaks_cleanup.py
python3 test_race_conditions_thread_safety.py

# End-to-end validation
python3 test_end_to_end_workflow.py
python3 validation_suite.py
```

## Development Utilities

### Git Operations
```bash
git status
git add .
git commit -m "descriptive message"
git log --oneline -10
```

### System Utilities (Linux/WSL)
```bash
# File operations
ls -la
find . -name "*.py"
grep -r "pattern" arena_bot/

# Process management
ps aux | grep python
top
```

### Python Environment
```bash
# Check Python version
python3 --version
which python3

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_windows.txt
pip install -r requirements_tier_integration.txt
```

## Debugging & Diagnostics
```bash
# Debug components
python3 debug_detection.py
python3 debug_coordinates.py
python3 diagnose_detection.py

# Visual debugging
python3 visual_debugger.py
python3 interactive_coordinate_finder.py
```