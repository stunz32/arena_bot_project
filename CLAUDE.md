# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Hearthstone Arena Draft Assistant** that helps players make optimal card choices during Arena drafts using computer vision and AI. The system achieves 100% accuracy in card detection and identification through sophisticated image processing and database filtering.

### Key Features
- Real-time card detection during Arena drafts
- AI-powered tier scoring and recommendations  
- Cross-platform support (Windows native, Linux/WSL)
- Visual overlay system for real-time guidance
- Underground mode support with redraft functionality

## Development Commands

### Running the Application

**Linux/WSL:**
```bash
# Main application entry point
python3 main.py

# Main GUI application
python3 integrated_arena_bot_gui.py
```

**Windows:**
```cmd
# One-time setup
SETUP_WINDOWS.bat

# Start the GUI application
START_ARENA_BOT_WINDOWS.bat
```

### Testing Commands

```bash
# Main test suite
python3 -m pytest tests/

# Core component testing
python3 test_core_components.py
python3 test_detection_accuracy.py

# Performance validation
python3 test_performance_bottlenecks.py
python3 test_ultimate_performance.py

# End-to-end workflow validation
python3 test_end_to_end_workflow.py
python3 validation_suite.py
```

### Development Dependencies

**Core Installation:**
```bash
pip install -r requirements.txt
```

**Platform-Specific:**
```bash
# Windows-specific dependencies
pip install -r requirements_windows.txt

# Testing dependencies  
pip install -r requirements-test.txt

# AI/ML components
pip install -r requirements_ai_v2.txt
```

## Architecture Overview

### Core Package Structure (`arena_bot/`)

**Core Components (`arena_bot/core/`):**
- `card_recognizer.py` - Main detection orchestrator and entry point
- `smart_coordinate_detector.py` - Card position detection (100% accuracy)
- `screen_detector.py` - Game state and screen detection
- `window_detector.py` - Hearthstone window management
- `thread_safe_state.py` - Thread-safe state management across components

**Detection Algorithms (`arena_bot/detection/`):**
- `histogram_matcher.py` - Arena Tracker-style histogram matching for card identification
- `template_matcher.py` - Mana cost and rarity detection using template matching
- `validation_engine.py` - Accuracy validation and confidence scoring
- `ultimate_detector.py` - Combined detection pipeline orchestrator

**Data Management (`arena_bot/data/`):**
- `arena_card_database.py` - Card database management and loading
- `card_eligibility_filter.py` - Smart database filtering (reduces DB by 80-85%)
- `heartharena_tier_manager.py` - External tier score integration
- `cards_json_loader.py` - JSON card data parsing and validation

**User Interface (`arena_bot/gui/`, `arena_bot/ui/`):**
- `draft_overlay.py` - Visual overlay for displaying recommendations
- `visual_overlay.py` - Real-time display components and graphics
- `settings_dialog.py` - Configuration and preferences GUI

**AI & Analysis (`arena_bot/ai/`, `arena_bot/ai_v2/`):**
- `draft_advisor.py` - Card recommendation engine
- `grandmaster_advisor.py` - Advanced AI analysis and explanations
- `conversational_coach.py` - Natural language card analysis

### Key Architecture Patterns

**Detection Pipeline:**
1. `smart_coordinate_detector.py` identifies card positions (100% accuracy)
2. `card_eligibility_filter.py` reduces database size by 80-85%
3. `histogram_matcher.py` performs Arena Tracker-style card matching
4. `template_matcher.py` validates mana costs and rarity
5. `validation_engine.py` ensures accuracy through confidence scoring

**Thread Safety:**
- All components use `thread_safe_state.py` for coordinated state management
- Async/await patterns throughout for performance
- Thread-safe logging via `arena_bot/logging_system/`

**Configuration System:**
- `bot_config.json` - Runtime configuration
- `arena_bot_logging_config.toml` - Comprehensive logging setup
- `arena_bot/utils/config.py` - Configuration loading and validation

## Code Style Guidelines

### Documentation Standards
- **JSDoc3 Format**: Use `/** ... */` for all multi-line comments
- **Focus on Intent**: Comment the "why" and "how", not obvious "what"
- **Required Documentation**:
  - All public APIs and exported functions
  - Complex algorithms and business logic
  - Non-obvious design decisions and trade-offs
  - Class and method docstrings with parameter descriptions

### Python Conventions
- **Type Hints**: Use modern Python typing throughout
- **Async Patterns**: Leverage async/await for performance-critical code
- **Error Handling**: Comprehensive exception handling with detailed logging
- **Naming**: 
  - Classes: `PascalCase` (e.g., `CardRecognizer`)
  - Functions/methods: `snake_case` (e.g., `detect_cards`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `CARD_POSITIONS`)
  - Private members: Leading underscore (e.g., `_card_recognizer`)

### Design Philosophy
- **Minimal Changes**: Preserve existing functionality when enhancing
- **Evidence-Based**: All modifications backed by testing and validation
- **Modular Design**: Clear separation of concerns across packages
- **Comprehensive Logging**: Log every logical workflow and decision point

## Technology Stack

**Core Dependencies:**
- **GUI Framework**: PyQt6 (6.6.0) for modern interface
- **Computer Vision**: OpenCV (4.8.1.78) for image processing
- **Image Handling**: Pillow (10.0.0) for screenshots and manipulation
- **Numerical Computing**: NumPy (1.24.3) for matrix operations
- **Machine Learning**: LightGBM (4.1.0) for AI recommendations

**Development Tools:**
- **Testing**: pytest (7.4.3) with pytest-cov for coverage
- **Performance**: memory-profiler, line-profiler for optimization
- **Data Validation**: jsonschema for configuration validation

## Development Workflow

### Making Changes
1. **Understand Current System**: Use existing logging and validation to understand behavior
2. **Preserve Functionality**: Maintain 100% accuracy in card detection
3. **Test Thoroughly**: Run detection accuracy tests before committing
4. **Document Changes**: Update comments following JSDoc3 standards
5. **Validate Performance**: Ensure changes don't impact real-time performance requirements

### Testing Strategy
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Validate component interactions
- **Accuracy Tests**: Maintain 100% card detection accuracy
- **Performance Tests**: Monitor real-time detection speed
- **End-to-End Tests**: Validate complete workflow from screenshot to recommendation

### Key Performance Targets
- **Card Detection**: 100% accuracy maintained
- **Real-time Performance**: Sub-second detection and recommendation
- **Database Efficiency**: 80-85% reduction through smart filtering
- **Memory Usage**: Efficient template and histogram caching