# Task Completion Workflow

## Standard Development Workflow

### 1. Code Quality Checks
**No automated linting/formatting tools configured** - manual code review required:
- Follow CLAUDE.md coding standards
- Ensure JSDoc3 style comments for all public APIs
- Verify comprehensive logging of workflows
- Check type hints and async/await patterns

### 2. Testing Requirements
```bash
# Run core functionality tests
python3 test_detection_accuracy.py

# Run comprehensive test suite
python3 -m pytest tests/ -v

# Platform-specific testing
# Windows:
run_test.bat
run_tier_tests.bat

# Linux/WSL:
./run_tier_tests.sh
python3 validation_suite.py
```

### 3. Performance Validation
```bash
# Check for performance regressions
python3 test_ultimate_performance.py
python3 test_performance_bottlenecks.py

# Memory leak detection
python3 test_memory_leaks_cleanup.py

# Thread safety validation
python3 test_race_conditions_thread_safety.py
```

### 4. Integration Testing
```bash
# End-to-end workflow validation
python3 test_end_to_end_workflow.py

# Component integration
python3 test_core_components.py
python3 test_tier_integration.py
```

### 5. Error Handling Verification
```bash
# Bulletproof fixes validation
python3 test_bulletproof_fixes.py
python3 test_error_fixes.py

# Chaos scenario testing
python3 test_chaos_scenarios.py
```

### 6. Manual Testing Steps
- Test GUI launches correctly on target platform
- Verify card detection accuracy with screenshots
- Check logging output for completeness
- Validate configuration loading

### 7. Documentation Updates
- Update relevant markdown files if architecture changes
- Ensure CLAUDE.md compliance for new code
- Update PROJECT_SUMMARY_COMPLETE.md for major changes

## Pre-Commit Checklist
- [ ] All tests pass
- [ ] No performance regressions
- [ ] Memory leaks checked
- [ ] Thread safety verified
- [ ] GUI functionality tested
- [ ] Logging output reviewed
- [ ] Comments and documentation updated
- [ ] CLAUDE.md standards followed

## Deployment Considerations
- Windows: Test with `START_ARENA_BOT_WINDOWS.bat`
- Linux/WSL: Test with `python3 main.py` and GUI components
- Verify cross-platform compatibility for any UI changes
- Check asset loading and database access across platforms