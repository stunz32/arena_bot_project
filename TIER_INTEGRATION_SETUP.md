# 🎯 HearthArena Tier Integration Setup Guide

## ❗ Required Dependencies

The new tier integration features require additional Python packages. You need to install these **before** running the tier-enabled bots.

## 🚀 Quick Installation

### Option 1: Automatic Installation (Recommended)

**Windows:**
```cmd
install_tier_dependencies.bat
```

**Linux/WSL:**
```bash
./install_tier_dependencies.sh
```

### Option 2: Manual Installation

Run these commands in your terminal/command prompt:

```bash
pip install beautifulsoup4 requests rapidfuzz lxml
```

Or if you're using Python 3 specifically:
```bash
pip3 install beautifulsoup4 requests rapidfuzz lxml
```

### Option 3: Using Requirements File

```bash
pip install -r requirements_tier_integration.txt
```

## 📦 What Each Package Does

| Package | Purpose | Why It's Needed |
|---------|---------|-----------------|
| **beautifulsoup4** | HTML parsing | Scrapes HearthArena tier lists (EzArena method) |
| **requests** | HTTP requests | Downloads tier data from HearthArena |
| **rapidfuzz** | Fuzzy string matching | Matches card names with database entries |
| **lxml** | Fast XML/HTML parsing | Speeds up BeautifulSoup parsing (optional but recommended) |

## ✅ Verify Installation

After installing, run this to test everything:

```bash
python test_tier_integration.py
```

You should see:
```
✅ PASS HearthArena Tier Manager
✅ PASS Tier Cache Manager  
✅ PASS Arena Database Integration
🎉 ALL TESTS PASSED!
```

## 🔧 Troubleshooting

### "Beautiful Soup not available" Error
```bash
pip install beautifulsoup4
```

### "rapidfuzz not available" Warning
```bash
pip install rapidfuzz
```
*Note: This is just a warning - fuzzy matching will use a slower fallback method*

### Permission Errors
Try installing with user flag:
```bash
pip install --user beautifulsoup4 requests rapidfuzz lxml
```

### Virtual Environment
If using a virtual environment, activate it first:
```bash
# Windows
venv\Scripts\activate
pip install beautifulsoup4 requests rapidfuzz lxml

# Linux/WSL  
source venv/bin/activate
pip install beautifulsoup4 requests rapidfuzz lxml
```

## 🎮 After Installation

Once dependencies are installed, you can run:

1. **Test Suite**: `python test_tier_integration.py`
2. **Enhanced Bot**: `python enhanced_arena_bot_with_tiers.py`
3. **Existing Bots**: All your existing bots will automatically get tier integration!

## 📝 Notes

- **Existing bots still work** without these packages (they just won't have tier data)
- **Tier integration is automatic** once packages are installed
- **First run downloads** fresh tier data (~2-3 minutes)
- **Subsequent runs use cache** (sub-second loading)

## 🌐 Network Requirements

The tier integration needs internet access to:
- Download arena rotation data (Arena Tracker URLs)
- Scrape tier lists from HearthArena.com
- This only happens during updates (every 24 hours for tiers, 7 days for arena data)

## 🎯 Ready to Go!

After installing dependencies, your arena bot will have:
- ✅ Arena eligibility filtering (Arena Tracker method)
- ✅ HearthArena tier rankings (EzArena method)  
- ✅ Binary caching for 10x+ performance
- ✅ Dual recommendation system (eligibility + tiers)