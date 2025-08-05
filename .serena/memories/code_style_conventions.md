# Code Style & Conventions

## Code Documentation Standards
- **JSDoc3 Style**: All multi-line comments use JSDoc format (`/** ... */`)
- **Comment Philosophy**: Focus on "why" and "how" rather than "what"
- **Required Comments**: 
  - Public APIs and exported functions
  - Complex algorithms and business logic
  - Non-obvious design decisions
  - Class and method docstrings
- **Avoid**: Obvious comments that restate code functionality

## Python Style Guidelines
- **Type Hints**: Modern Python typing throughout codebase
- **Async/Await**: Extensive use of async patterns for performance
- **Class Organization**: Clear separation of concerns with modular design
- **Error Handling**: Comprehensive exception handling with logging
- **Constants**: Upper case naming (e.g., `CARD_POSITIONS`, `MANA_REGIONS`)

## File Organization
- **Package Structure**: Modular arena_bot package with subpackages:
  - `core/` - Core detection and recognition logic
  - `gui/` - User interface components  
  - `utils/` - Shared utilities and configuration
  - `data/` - Database and data management
  - `detection/` - Computer vision algorithms
  - `logging_system/` - Advanced logging infrastructure

## Naming Conventions
- **Classes**: PascalCase (e.g., `CardRecognizer`, `HistogramMatcher`)
- **Functions/Methods**: snake_case (e.g., `detect_cards`, `extract_card_regions`)
- **Variables**: snake_case (e.g., `detection_stats`, `is_initialized`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CARD_POSITIONS`)
- **Private Members**: Leading underscore (e.g., `_card_recognizer`)

## Project Philosophy
- **Clarity and Maintainability**: Primary goal for all code
- **Evidence-Based Development**: All changes backed by testing
- **Minimal Changes**: Preserve existing functionality when enhancing
- **Comprehensive Logging**: Log every logical connection and workflow