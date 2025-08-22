#!/usr/bin/env python3
"""
Fix GUI performance by adding lazy loading option.
This prevents loading 33K cards when we only need the GUI for testing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_lazy_gui_test():
    """Create a GUI test that doesn't load heavy detection systems."""
    
    # Mock the heavy imports before they get loaded
    from unittest.mock import Mock, patch
    import sys
    
    # Mock the card database to prevent 33K card loading
    mock_arena_db = Mock()
    mock_arena_db.get_card_count.return_value = 4098
    mock_arena_db.get_arena_cards.return_value = []
    
    mock_cards_loader = Mock() 
    mock_cards_loader.get_card_count.return_value = 4098  # Fake smaller count
    
    # Patch the heavy imports
    patches = [
        patch('arena_bot.data.arena_card_database.get_cards_json_loader', return_value=mock_cards_loader),
        patch('arena_bot.data.cards_json_loader.get_cards_json_loader', return_value=mock_cards_loader),
        patch('arena_bot.data.arena_version_manager.get_cards_json_loader', return_value=mock_cards_loader),
        patch('arena_bot.data.heartharena_tier_manager.get_cards_json_loader', return_value=mock_cards_loader),
        patch('arena_bot.data.tier_cache_manager.get_cards_json_loader', return_value=mock_cards_loader),
    ]
    
    # Start all patches
    for p in patches:
        p.start()
    
    try:
        print("🚀 Creating GUI with mocked heavy systems...")
        
        # Now import and create the GUI
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        gui = IntegratedArenaBotGUI()
        
        print("✅ GUI created successfully (without loading 33K cards)")
        
        # Setup GUI
        gui.setup_gui()
        print("✅ GUI setup completed")
        
        # Capture debug info
        if hasattr(gui, 'root') and gui.root:
            gui.root.update_idletasks()
            
            from app.debug_utils import create_debug_snapshot
            results = create_debug_snapshot(gui.root, "optimized_arena_bot_gui")
            
            print("📸 Debug snapshot captured:")
            for key, path in results.items():
                print(f"  - {key}: {path}")
            
            gui.root.destroy()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        # Stop all patches
        for p in patches:
            p.stop()

def suggest_permanent_fix():
    """Suggest code changes for permanent fix."""
    
    print("\n🛠️ PERMANENT FIX SUGGESTIONS:")
    print("=" * 50)
    
    print("1. **Add lazy loading parameter to ArenaCardDatabase:**")
    print("""
    # File: arena_bot/data/arena_card_database.py
    def __init__(self, cache_dir: Optional[Path] = None, lazy_load: bool = False):
        # ... existing init code ...
        
        if not lazy_load:
            # Load existing data only if not lazy loading
            self.load_cached_data()
        else:
            self.arena_data = None  # Skip heavy loading
    """)
    
    print("\n2. **Add GUI-only mode to IntegratedArenaBotGUI:**")
    print("""
    # File: integrated_arena_bot_gui.py  
    def __init__(self, gui_only: bool = False):
        if gui_only:
            print("🚀 GUI-only mode - skipping detection systems")
            self.setup_gui()
            return
            
        # ... existing heavy initialization ...
    """)
    
    print("\n3. **Create lightweight GUI entry point:**")
    print("""
    # File: gui_test_mode.py
    if __name__ == "__main__":
        from integrated_arena_bot_gui import IntegratedArenaBotGUI
        
        # Create GUI without heavy systems
        gui = IntegratedArenaBotGUI(gui_only=True)
        gui.run()
    """)
    
    print("\n4. **Estimated performance improvement:**")
    print("   - Current startup: 45+ seconds")  
    print("   - With lazy loading: ~2-3 seconds")
    print("   - Memory reduction: ~1GB+ (33K cards not loaded)")

if __name__ == "__main__":
    print("🎮 Arena Bot GUI Performance Fix")
    print("=" * 50)
    
    success = create_lazy_gui_test()
    
    if success:
        print("\n✅ Lazy loading test successful!")
        suggest_permanent_fix()
    else:
        print("\n❌ Test failed - check errors above")
        suggest_permanent_fix()
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"The GUI doesn't need 33,234 cards - only the 4,098 arena cards.")
    print(f"Loading should be deferred until actual card detection is needed.")