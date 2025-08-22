# Test Fixtures

This directory contains test fixtures for Arena Bot testing.

## End-to-End Fixtures

### `end_to_end/drafts/`

Contains synthetic Arena draft screenshots with sidecar labels for offline replay testing.

**Sidecar Label Format:**
- Each PNG file can have a corresponding `.labels.json` file
- Sidecar files enable offline replay without actual computer vision processing
- Format: `{"cards": [{"id": "...", "name": "...", "mana_cost": N, "tier_score": N.N}, ...]}`

**Current Fixtures:**
- `pack_001.png` + `pack_001.labels.json` - Three cards with A > B > C tier scoring
- `pack_002.png` + `pack_002.labels.json` - Three cards with different tier ordering

**Replacement:**
These synthetic fixtures can be replaced with real Hearthstone screenshots and corresponding labels extracted from actual Arena Tracker detection results.

## Usage

```python
from arena_bot.cli import run_replay

# Process all fixtures
results = run_replay("tests/fixtures/end_to_end/drafts", offline=True, debug_tag="test")

# Access per-frame results
for result in results:
    cards = result["cards"]  # Sorted by tier_score descending
    print(f"Top pick: {cards[0]['name']} (score: {cards[0]['tier_score']})")
```