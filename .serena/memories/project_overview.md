# Arena Bot Project Overview

## Project Purpose
The Arena Bot is a **Hearthstone Arena Draft Assistant** that helps players make optimal card choices during Arena drafts. It uses computer vision and AI to:

- Detect cards on screen during Arena drafts
- Provide tier scores and recommendations for card picks
- Support both regular Arena mode and Underground mode with redraft functionality
- Offer real-time analysis with explanations for card choices

## Current Status
- ✅ **100% accuracy achieved** for card detection and identification
- ✅ Complete card recognition system implemented
- ✅ Cross-platform support (Windows native, Linux/WSL)
- ✅ GUI interface with real-time monitoring
- ⚠️ AI components recently integrated (some logging issues present)

## Key Features
- **Smart Coordinate Detection**: 100% accuracy in detecting card positions
- **Histogram Matching**: Arena Tracker-style card identification
- **Template Matching**: Mana cost and rarity detection
- **Database Filtering**: Smart pre-filtering reduces database size by 80-85%
- **Visual Overlay**: Real-time display of recommendations
- **Cross-Platform**: Native Windows support + Linux/WSL compatibility

## Architecture
Built with modern Python architecture using:
- Modular design with clear separation of concerns
- Async/await patterns for performance
- Thread-safe components for concurrent operations
- Comprehensive logging system
- Validation engines for accuracy assurance