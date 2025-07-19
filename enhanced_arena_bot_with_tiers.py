#!/usr/bin/env python3
"""
Enhanced Arena Bot with HearthArena Tier Integration

Production-ready arena bot featuring EzArena-style tier integration.
Combines Arena Tracker filtering with HearthArena tier rankings.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the arena_bot module to path
sys.path.insert(0, str(Path(__file__).parent / "arena_bot"))

class EnhancedArenaBotWithTiers:
    """
    Enhanced Arena Bot with integrated HearthArena tier data.
    
    Features:
    - Arena Tracker's proven eligibility filtering
    - EzArena's HearthArena tier scraping approach
    - Binary tier caching for 10x+ performance
    - Dual recommendation system (eligibility + tiers)
    """
    
    def __init__(self):
        """Initialize the enhanced arena bot."""
        self.logger = logging.getLogger(__name__)
        
        # Load all required components
        self.load_components()
        
        # Initialize data
        self.initialize_data()
        
        self.logger.info("✅ Enhanced Arena Bot with Tiers initialized")
    
    def load_components(self):
        """Load all required arena bot components."""
        try:
            from arena_bot.data.arena_card_database import get_arena_card_database
            from arena_bot.data.arena_version_manager import get_arena_version_manager
            from arena_bot.data.heartharena_tier_manager import get_heartharena_tier_manager
            from arena_bot.data.tier_cache_manager import get_tier_cache_manager
            from arena_bot.data.cards_json_loader import get_cards_json_loader
            
            self.arena_db = get_arena_card_database()
            self.version_manager = get_arena_version_manager()
            self.tier_manager = get_heartharena_tier_manager()
            self.tier_cache = get_tier_cache_manager()
            self.cards_loader = get_cards_json_loader()
            
            self.logger.info("✅ All components loaded successfully")
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to load components: {e}")
            raise
    
    def initialize_data(self):
        """Initialize and update all data sources."""
        self.logger.info("🚀 Initializing arena data...")
        
        # Check if arena database needs updating
        needs_update, reason = self.arena_db.check_for_updates()
        if needs_update:
            self.logger.info(f"📥 Updating arena database: {reason}")
            success = self.arena_db.update_from_arena_version(force=False)
            if not success:
                self.logger.warning("⚠️ Arena database update failed, using cached data")
        
        # Ensure tier integration is current
        self.arena_db.update_with_tier_data(force=False)
        
        # Show initialization summary
        self.show_initialization_summary()
    
    def show_initialization_summary(self):
        """Show a summary of the initialized data."""
        print("\n" + "="*60)
        print("🎯 ENHANCED ARENA BOT INITIALIZATION SUMMARY")
        print("="*60)
        
        # Arena database info
        db_info = self.arena_db.get_database_info()
        print(f"📊 Arena Database:")
        print(f"   Status: {db_info['status']}")
        print(f"   Total arena cards: {db_info['total_cards']}")
        print(f"   Cache age: {db_info['cache_age_days']:.1f} days")
        print(f"   Source: {db_info['source']}")
        
        # Class card counts
        print(f"\n📋 Cards by Class:")
        for class_name, count in db_info['card_counts'].items():
            print(f"   {class_name.title()}: {count} cards")
        
        # Tier integration status
        tier_stats = db_info.get('tier_stats', {})
        if tier_stats.get('has_tier_data'):
            print(f"\n🎯 HearthArena Tier Integration:")
            print(f"   Status: ✅ ACTIVE")
            print(f"   Classes with tiers: {tier_stats['classes_with_tiers']}")
            print(f"   Cards with tier data: {tier_stats['total_cards_with_tiers']}")
            
            print(f"\n   Tier Distribution:")
            for tier, count in tier_stats.get('tier_distribution', {}).items():
                print(f"     {tier}: {count} cards")
        else:
            print(f"\n🎯 HearthArena Tier Integration:")
            print(f"   Status: ❌ NOT AVAILABLE")
        
        # Cache performance
        cache_info = db_info.get('tier_cache_info', {})
        if cache_info.get('status') == 'loaded':
            print(f"\n⚡ Cache Performance:")
            print(f"   Cache size: {cache_info['cache_size_bytes']:,} bytes")
            print(f"   Compression: {cache_info.get('compression_ratio', 1.0):.1f}x")
            if 'performance' in cache_info:
                perf = cache_info['performance']
                print(f"   Save time: {perf['save_time_ms']:.1f}ms")
                print(f"   Efficiency: {perf['compression_efficiency']:.1f}%")
        
        print("="*60)
    
    def get_card_recommendations(self, card_names: List[str], hero_class: str) -> List[Dict]:
        """
        Get enhanced card recommendations with tier data.
        
        Args:
            card_names: List of card names to evaluate
            hero_class: Hero class for the draft
            
        Returns:
            List of card recommendations with tier and eligibility info
        """
        recommendations = []
        
        for card_name in card_names:
            recommendation = {
                'card_name': card_name,
                'arena_eligible': False,
                'tier_data': None,
                'recommendation_score': 0,
                'reasons': []
            }
            
            # Check arena eligibility (by card ID)
            card_matches = self.cards_loader.get_card_matches_fuzzy(card_name)
            if card_matches:
                card_id = card_matches[0].card_id
                recommendation['card_id'] = card_id
                
                is_eligible = self.arena_db.is_card_arena_eligible(card_id, hero_class.lower())
                recommendation['arena_eligible'] = is_eligible
                
                if is_eligible:
                    recommendation['recommendation_score'] += 50
                    recommendation['reasons'].append("Arena eligible")
                else:
                    recommendation['reasons'].append("NOT arena eligible")
            else:
                recommendation['reasons'].append("Card not found in database")
            
            # Get tier data (by card name)
            tier_data = self.arena_db.get_card_tier_fast(card_name, hero_class.lower())
            if tier_data:
                recommendation['tier_data'] = {
                    'tier': tier_data.tier,
                    'tier_index': tier_data.tier_index,
                    'confidence': tier_data.confidence
                }
                
                # Score based on tier (lower tier_index = better)
                tier_score = (7 - tier_data.tier_index) * 10  # 70 for best, 0 for worst
                recommendation['recommendation_score'] += tier_score
                recommendation['reasons'].append(f"HearthArena tier: {tier_data.tier}")
            else:
                recommendation['reasons'].append("No tier data available")
            
            recommendations.append(recommendation)
        
        # Sort by recommendation score (highest first)
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return recommendations
    
    def format_card_recommendation(self, rec: Dict) -> str:
        """Format a card recommendation for display."""
        card_name = rec['card_name']
        score = rec['recommendation_score']
        
        # Status indicators
        eligible_icon = "✅" if rec['arena_eligible'] else "❌"
        
        tier_info = ""
        if rec['tier_data']:
            tier = rec['tier_data']['tier']
            tier_index = rec['tier_data']['tier_index']
            
            # Tier icons
            tier_icons = {
                'beyond-great': '🔥', 'great': '⭐', 'good': '👍',
                'above-average': '🙂', 'average': '😐', 'below-average': '👎',
                'bad': '💀', 'terrible': '☠️'
            }
            tier_icon = tier_icons.get(tier, '❓')
            tier_info = f" {tier_icon} {tier}"
        
        reasons = " | ".join(rec['reasons'])
        
        return f"{eligible_icon} {card_name} (Score: {score}){tier_info} - {reasons}"
    
    def demo_card_evaluation(self):
        """Demonstrate card evaluation with example cards."""
        print("\n" + "="*60)
        print("🎮 DEMO: CARD EVALUATION WITH TIER DATA")
        print("="*60)
        
        # Example draft scenarios
        scenarios = [
            {
                'hero_class': 'mage',
                'cards': ['Fireball', 'Frostbolt', 'Arcane Intellect', 'Flamestrike', 'Polymorph']
            },
            {
                'hero_class': 'warrior',
                'cards': ['Execute', 'Fiery War Axe', 'Shield Slam', 'Brawl', 'Armorsmith']
            },
            {
                'hero_class': 'hunter',
                'cards': ['Animal Companion', 'Kill Command', 'Unleash the Hounds', 'Tracking', 'Hunter\'s Mark']
            }
        ]
        
        for scenario in scenarios:
            hero_class = scenario['hero_class']
            cards = scenario['cards']
            
            print(f"\n🎯 {hero_class.title()} Draft Scenario:")
            print(f"Cards to evaluate: {', '.join(cards)}")
            print(f"\nRecommendations (best to worst):")
            
            recommendations = self.get_card_recommendations(cards, hero_class)
            
            for i, rec in enumerate(recommendations, 1):
                formatted = self.format_card_recommendation(rec)
                print(f"  {i}. {formatted}")
        
        print("\n" + "="*60)
        print("Legend:")
        print("✅ = Arena eligible  ❌ = Not arena eligible")
        print("🔥 = Beyond Great  ⭐ = Great  👍 = Good  🙂 = Above Average")
        print("😐 = Average  👎 = Below Average  💀 = Bad  ☠️ = Terrible")
        print("="*60)
    
    def show_tier_statistics(self):
        """Show comprehensive tier statistics."""
        print("\n" + "="*60)
        print("📊 HEARTHARENA TIER STATISTICS")
        print("="*60)
        
        # Get tier cache statistics
        cache_stats = self.tier_cache.get_cache_statistics()
        
        if cache_stats['status'] == 'loaded':
            print(f"✅ Tier data loaded successfully")
            print(f"Total tier entries: {cache_stats['total_entries']}")
            print(f"Classes with tiers: {cache_stats['classes_cached']}")
            print(f"Cache size: {cache_stats['cache_size_bytes']:,} bytes")
            print(f"Compression ratio: {cache_stats['compression_ratio']:.1f}x")
            print(f"Cache age: {cache_stats['cache_age_hours']:.1f} hours")
            
            # Show tier distribution by class
            print(f"\n📋 Tier Distribution by Class:")
            for class_name in ['mage', 'warrior', 'hunter', 'priest', 'warlock', 'rogue', 'shaman', 'paladin', 'druid', 'demon-hunter']:
                class_tiers = self.tier_cache.get_class_tiers(class_name)
                if class_tiers:
                    tier_counts = {}
                    for tier_data in class_tiers.values():
                        tier = tier_data.tier
                        tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    
                    print(f"\n{class_name.title()}: {len(class_tiers)} cards")
                    for tier in ['beyond-great', 'great', 'good', 'above-average', 'average', 'below-average', 'bad', 'terrible']:
                        count = tier_counts.get(tier, 0)
                        if count > 0:
                            print(f"  {tier}: {count} cards")
        else:
            print(f"❌ No tier data available")
        
        print("="*60)
    
    def interactive_mode(self):
        """Run the bot in interactive mode."""
        print("\n" + "="*60)
        print("🎮 INTERACTIVE MODE")
        print("="*60)
        print("Enter card names to get tier-enhanced recommendations!")
        print("Commands:")
        print("  'quit' or 'exit' - Exit interactive mode")
        print("  'stats' - Show tier statistics")
        print("  'demo' - Run demo scenarios")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter hero class (e.g., mage): ").strip().lower()
                if user_input in ['quit', 'exit']:
                    break
                elif user_input == 'stats':
                    self.show_tier_statistics()
                    continue
                elif user_input == 'demo':
                    self.demo_card_evaluation()
                    continue
                
                if not user_input:
                    continue
                
                hero_class = user_input
                
                # Get card names
                cards_input = input(f"Enter card names for {hero_class} (comma-separated): ").strip()
                if not cards_input:
                    continue
                
                card_names = [name.strip() for name in cards_input.split(',')]
                
                # Get recommendations
                print(f"\n🎯 Recommendations for {hero_class.title()}:")
                recommendations = self.get_card_recommendations(card_names, hero_class)
                
                for i, rec in enumerate(recommendations, 1):
                    formatted = self.format_card_recommendation(rec)
                    print(f"  {i}. {formatted}")
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def run(self):
        """Run the enhanced arena bot."""
        print("\n🎯 Enhanced Arena Bot with HearthArena Tiers")
        print("Combining Arena Tracker filtering with EzArena tier scraping!")
        
        # Show what we can do
        print("\n📋 Available Functions:")
        print("1. Demo card evaluation scenarios")
        print("2. Show tier statistics")
        print("3. Interactive card recommendation mode")
        
        while True:
            try:
                choice = input("\nChoose an option (1-3) or 'quit': ").strip()
                
                if choice.lower() in ['quit', 'exit']:
                    break
                elif choice == '1':
                    self.demo_card_evaluation()
                elif choice == '2':
                    self.show_tier_statistics()
                elif choice == '3':
                    self.interactive_mode()
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 'quit'.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break

def setup_logging():
    """Setup logging for the bot."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_arena_bot.log')
        ]
    )

def main():
    """Main entry point."""
    print("🎯 ENHANCED ARENA BOT WITH HEARTHARENA TIER INTEGRATION")
    print("=" * 80)
    print("This bot demonstrates the new tier integration features:")
    print("• Arena Tracker's proven eligibility filtering")
    print("• EzArena's HearthArena tier scraping approach") 
    print("• Binary tier caching for 10x+ performance")
    print("• Dual recommendation system (eligibility + tiers)")
    print("=" * 80)
    
    setup_logging()
    
    try:
        bot = EnhancedArenaBotWithTiers()
        bot.run()
    except Exception as e:
        print(f"❌ Failed to start enhanced arena bot: {e}")
        logging.exception("Bot startup failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)