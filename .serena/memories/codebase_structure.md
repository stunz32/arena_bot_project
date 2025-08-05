# Codebase Structure

## Main Entry Points
- `main.py` - Primary application entry point with initialization
- `enhanced_realtime_arena_bot.py` - Real-time GUI bot for Linux/WSL
- `START_ARENA_BOT_WINDOWS.bat` - Windows native launcher

## Core Package Structure (`arena_bot/`)

### Core Components (`arena_bot/core/`)
- `card_recognizer.py` - Main card detection orchestrator
- `screen_detector.py` - Screen state detection
- `smart_coordinate_detector.py` - Card position detection (100% accuracy)
- `window_detector.py` - Hearthstone window management
- `thread_safe_state.py` - Thread-safe state management

### Detection Algorithms (`arena_bot/detection/`)
- `histogram_matcher.py` - Arena Tracker-style histogram matching
- `template_matcher.py` - Mana cost and rarity detection
- `validation_engine.py` - Accuracy validation system
- `ultimate_detector.py` - Combined detection pipeline
- `phash_matcher.py` - Perceptual hash matching

### Data Management (`arena_bot/data/`)
- `arena_card_database.py` - Card database management
- `card_eligibility_filter.py` - Smart database filtering (83% reduction)
- `heartharena_tier_manager.py` - Tier score integration
- `cards_json_loader.py` - JSON card data loading

### User Interface (`arena_bot/gui/`, `arena_bot/ui/`)
- `draft_overlay.py` - Visual overlay for recommendations
- `settings_dialog.py` - Configuration GUI
- `visual_overlay.py` - Real-time display components

### AI & Analysis (`arena_bot/ai/`, `arena_bot/ai_v2/`)
- `draft_advisor.py` - Card recommendation engine
- `grandmaster_advisor.py` - Advanced AI analysis
- `conversational_coach.py` - Natural language explanations

### Advanced Logging (`arena_bot/logging_system/`)
- Comprehensive async logging infrastructure
- Hierarchical configuration system
- Performance monitoring and diagnostics

### Utilities (`arena_bot/utils/`)
- `config.py` - Configuration management
- `logging_config.py` - Logging setup
- `asset_loader.py` - Template and asset loading

## Test Structure
- `test_*.py` - Main test files (pytest-based)
- `tests/` - Organized test directory
- `test_files/` - Test assets and validation data

## Configuration Files
- `bot_config.json` - Runtime configuration
- `arena_bot_logging_config.toml` - Logging configuration
- `requirements*.txt` - Python dependencies for different platforms

## Documentation & Reports
- `PROJECT_SUMMARY_COMPLETE.md` - Complete project status
- `CLAUDE.md` - Development guidelines and standards
- Various checkpoint and analysis documents