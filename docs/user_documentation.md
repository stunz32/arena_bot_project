# Arena Bot AI v2 User Guide

## Welcome to Your Grandmaster AI Coach

The Arena Bot AI v2 transforms your Hearthstone Arena experience by providing expert-level guidance from hero selection through all 30 card picks. This comprehensive guide will help you understand and maximize the power of your new AI coach.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Hero Selection Strategy](#hero-selection-strategy)
3. [Understanding Card Recommendations](#understanding-card-recommendations)
4. [Statistical Interpretation](#statistical-interpretation)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements
- Windows 10/11 with Hearthstone installed
- Python 3.8 or higher
- Active internet connection for real-time data
- 4GB RAM minimum, 8GB recommended

### First Launch
1. Start Hearthstone and navigate to Arena mode
2. Launch Arena Bot AI v2
3. The system will automatically detect when you enter Arena draft
4. Wait for the "Ready" indicator in the top-right corner

### Interface Overview
The Arena Bot interface consists of several key areas:

- **Status Panel**: Shows connection status and data availability
- **Hero Selection Panel**: Appears when choosing your hero
- **Card Recommendation Panel**: Displays during card picks
- **Statistics Panel**: Shows real-time performance data
- **Coach Panel**: Provides strategic insights and explanations

## Hero Selection Strategy

### Understanding Hero Recommendations

When you're presented with three hero choices, the AI analyzes each option using current meta data and provides detailed recommendations:

#### Example Hero Analysis
```
┌─────────────────────────────────────────────────────────┐
│ HERO RECOMMENDATIONS (Meta Updated: 2 hours ago)       │
├─────────────────────────────────────────────────────────┤
│ 🥇 WARRIOR (Rank 1) - 56.5% Winrate                    │
│    Confidence: 92%                                     │
│    Playstyle: Aggressive/Control Hybrid                │
│    Complexity: Medium                                   │
│    Best Archetypes: Aggro (40%), Midrange (35%)        │
│                                                         │
│ 🥈 MAGE (Rank 2) - 54.5% Winrate                       │
│    Confidence: 89%                                     │
│    Playstyle: Spell-centric Control                    │
│    Complexity: High                                     │
│    Best Archetypes: Control (50%), Midrange (30%)      │
│                                                         │
│ 🥉 HUNTER (Rank 3) - 49.2% Winrate                     │
│    Confidence: 91%                                     │
│    Playstyle: Aggressive Face                          │
│    Complexity: Low                                      │
│    Best Archetypes: Aggro (70%), Midrange (30%)        │
└─────────────────────────────────────────────────────────┘
```

#### Key Metrics Explained

**Winrate**: Current meta performance based on thousands of Arena games
- **Above 55%**: Excellent choice, strong in current meta
- **50-55%**: Good choice, solid performance
- **45-50%**: Average choice, requires skill to pilot
- **Below 45%**: Challenging choice, not recommended for most players

**Confidence**: How reliable the recommendation is
- **90%+**: High confidence, recent and extensive data
- **80-90%**: Good confidence, sufficient data available
- **70-80%**: Moderate confidence, limited recent data
- **Below 70%**: Low confidence, using fallback estimates

**Complexity**: How difficult the hero is to pilot effectively
- **Low**: Straightforward strategy, forgiving of mistakes
- **Medium**: Requires understanding of key interactions
- **High**: Demands expert knowledge and precise play

### Hero Selection Tips

#### For New Players
- Choose heroes with **Low** or **Medium** complexity
- Prioritize heroes with winrates above 52%
- Avoid heroes requiring specific card synergies

#### For Experienced Players
- Consider **High** complexity heroes if you're familiar with their strategies
- Pay attention to archetype preferences that match your playstyle
- Factor in your personal performance history with each class

#### Meta Considerations
- **Trending Up** 📈: Hero performing better recently
- **Stable** ➡️: Consistent performance over time
- **Trending Down** 📉: Hero losing effectiveness in current meta

## Understanding Card Recommendations

### The Recommendation Display

During card picks, you'll see detailed analysis for each card option:

```
┌─────────────────────────────────────────────────────────┐
│ CARD PICK #7 - WARRIOR DRAFT                           │
├─────────────────────────────────────────────────────────┤
│ 🥇 Fiery War Axe (RECOMMENDED)                         │
│    Overall Score: 8.7/10                               │
│    Base Value: 8.2 | Synergy: 9.5 | Curve: 8.0        │
│    Reason: Excellent weapon synergy with Warrior       │
│    Meta Winrate: 58.3% (↑2.1% this week)              │
│                                                         │
│ 🥈 Consecration                                         │
│    Overall Score: 7.1/10                               │
│    Base Value: 7.8 | Synergy: 4.2 | Curve: 8.5        │
│    Reason: Strong AoE but poor class synergy           │
│    Meta Winrate: 54.7% (→ stable)                      │
│                                                         │
│ 🥉 Magma Rager                                          │
│    Overall Score: 3.2/10                               │
│    Base Value: 3.1 | Synergy: 3.0 | Curve: 3.5        │
│    Reason: Poor stats, high risk play                  │
│    Meta Winrate: 38.9% (↓1.3% this week)              │
└─────────────────────────────────────────────────────────┘
```

### Score Components Explained

#### Overall Score (1-10 scale)
The Overall Score combines all factors into a single recommendation:
- **9.0+**: Exceptional pick, take immediately
- **8.0-8.9**: Excellent pick, strong choice
- **7.0-7.9**: Good pick, solid addition to deck
- **6.0-6.9**: Average pick, acceptable if needed
- **5.0-5.9**: Below average, avoid if possible
- **Below 5.0**: Poor pick, only take if desperate

#### Base Value
The card's fundamental power level in Arena:
- Considers mana cost efficiency
- Evaluates attack/health stats
- Accounts for card text effects
- Based on thousands of game outcomes

#### Synergy Score
How well the card works with your chosen hero and current deck:
- **Hero Synergy**: Class-specific bonuses and interactions
- **Archetype Fit**: How well it matches your developing strategy
- **Existing Cards**: Synergies with already drafted cards
- **Combo Potential**: Future synergy possibilities

#### Curve Score
How well the card fits your deck's mana curve:
- Evaluates current mana distribution
- Considers hero-specific curve preferences
- Accounts for game tempo requirements
- Balances early, mid, and late game needs

### Meta Winrate Indicators

**Trending Indicators**:
- **↑**: Card performing better recently (gaining popularity)
- **→**: Stable performance over time
- **↓**: Card losing effectiveness (declining winrate)

**Winrate Interpretation**:
- **60%+**: Premium card, among the best in Arena
- **55-60%**: Excellent card, strong in current meta
- **50-55%**: Good card, solid performer
- **45-50%**: Average card, situationally useful
- **40-45%**: Below average, generally avoid
- **Below 40%**: Poor card, only in desperate situations

## Statistical Interpretation

### Understanding Confidence Levels

The AI provides confidence levels for all recommendations to help you understand data reliability:

#### High Confidence (90%+)
- **What it means**: Extensive recent data available
- **When you see it**: Popular cards and meta heroes
- **Trust level**: Follow recommendations closely
- **Example**: "Fiery War Axe - 94% confidence"

#### Good Confidence (80-90%)
- **What it means**: Sufficient data, reliable patterns
- **When you see it**: Most standard cards and heroes
- **Trust level**: Recommendations are very reliable
- **Example**: "Lightning Bolt - 86% confidence"

#### Moderate Confidence (70-80%)
- **What it means**: Limited recent data, using broader patterns
- **When you see it**: Newer cards or rarely picked options
- **Trust level**: Good guidance, consider your experience
- **Example**: "Obscure Epic Card - 74% confidence"

#### Low Confidence (Below 70%)
- **What it means**: Minimal data, using fallback systems
- **When you see it**: Very new cards or API issues
- **Trust level**: Use as rough guidance only
- **Example**: "Brand New Card - 62% confidence"

### Data Freshness Indicators

#### Real-Time Data Available 🟢
- Hero and card data updated within last 6 hours
- Full API connectivity
- Maximum recommendation accuracy

#### Recent Data Available 🟡
- Data updated within last 24 hours
- Possible minor API delays
- Excellent recommendation accuracy

#### Cached Data Only 🟠
- Using locally stored data (1-7 days old)
- API temporarily unavailable
- Good recommendation accuracy

#### Fallback Mode 🔴
- Using built-in tier lists and algorithms
- No external data connectivity
- Basic recommendation accuracy

### Personal Performance Tracking

The system learns from your draft choices and results:

#### Archetype Preference Learning
```
Your Warrior Performance:
┌─────────────────────────────────────────┐
│ Aggro Warrior:    7 runs, 65% winrate  │
│ Control Warrior:  3 runs, 45% winrate  │ 
│ Midrange Warrior: 5 runs, 73% winrate  │
└─────────────────────────────────────────┘
Recommendation: Focus on Midrange strategy
```

#### Card Performance Feedback
- The AI tracks which cards perform well in your hands
- Recommendations gradually adapt to your playstyle
- "Personal bias" indicators show when recommendations differ from meta

## Advanced Features

### Manual Override System

Sometimes you may want to pick differently than recommended:

#### How to Override
1. Right-click on your preferred card choice
2. Select "Override Recommendation"
3. Choose reason: "Personal preference", "Specific strategy", "Archetype pivot"
4. The AI will adapt future recommendations accordingly

#### Override Tracking
```
Override Summary (Last 10 Drafts):
┌─────────────────────────────────────────┐
│ Personal Preference: 15 overrides      │
│ Success Rate: 68% (above AI: 71%)      │
│ Recommendation: Trust AI slightly more │
└─────────────────────────────────────────┘
```

### Archetype Pivot Detection

The AI monitors your draft for archetype changes:

#### Pivot Alerts
```
🔄 ARCHETYPE PIVOT DETECTED
Current: Midrange Warrior (73% lean)
Suggested: Aggro Warrior (upcoming aggressive cards)
Reason: Strong early game synergies available
Confidence: 82%
```

#### Pivot Confirmation
- **Accept**: AI adjusts all future recommendations
- **Decline**: Continue with current archetype
- **Manual**: Take control of archetype decisions

### Draft Review and Export

After completing your draft, access comprehensive analysis:

#### Draft Summary
- Hero choice analysis and accuracy
- Card-by-card decision review
- Archetype development timeline
- Predicted performance range
- Key synergies identified
- Potential weaknesses

#### Export Options
- **Text Summary**: Copy to clipboard for sharing
- **Detailed Report**: Full statistical breakdown
- **Arena Tracker**: Compatible with popular tracking tools
- **Personal Database**: Store for future reference

### Conversational Coach

Ask your AI coach questions during the draft:

#### Example Questions
- "Why is this card recommended?"
- "What archetype am I building?"
- "Should I prioritize curve or value?"
- "What cards should I look for next?"
- "How does this card synergize with my hero?"

#### Coach Response Example
```
🤖 AI Coach Response:
"Fiery War Axe is recommended because:
1. Excellent value (3/2 weapon for 2 mana)
2. Strong Warrior synergy (+25% effectiveness)
3. Fills your 2-mana curve gap
4. Enables weapon-based strategies
5. Currently 58.3% winrate in meta

This pick improves your aggro/midrange potential
and gives early board control options."
```

## Tips for Maximum Success

### Draft Strategy

#### Early Picks (1-10)
- **Prioritize**: High base value cards and hero synergies
- **Avoid**: Narrow combo pieces or situational cards
- **Focus**: Building a solid foundation and establishing archetype

#### Mid Picks (11-20)
- **Prioritize**: Synergy cards and curve completion
- **Avoid**: Off-archetype cards unless exceptional value
- **Focus**: Strengthening your chosen strategy

#### Late Picks (21-30)
- **Prioritize**: Filling curve gaps and specific answers
- **Avoid**: Redundant effects or poor value
- **Focus**: Completing your game plan and covering weaknesses

### Reading Opponent Patterns

The AI can help you understand common Arena patterns:

#### Meta Insights
- "67% of Mages are playing spell-heavy decks this week"
- "Warrior weapons are 15% more common than usual"
- "Control strategies are underperforming (-3.2% winrate)"

#### Mulligan Guidance
Based on your final deck and meta knowledge:
```
Mulligan Guide vs. Common Matchups:
┌─────────────────────────────────────────┐
│ vs. Aggro: Keep 1-3 mana cards          │
│ vs. Control: Keep value engines         │
│ vs. Midrange: Keep efficient trades     │
└─────────────────────────────────────────┘
```

### Common Mistakes to Avoid

#### Over-relying on the AI
- **Problem**: Blindly following every recommendation
- **Solution**: Understand the reasoning behind picks
- **Tip**: Ask the AI coach "why" when uncertain

#### Ignoring Archetype Signals
- **Problem**: Picking good cards that don't fit your strategy
- **Solution**: Pay attention to archetype lean percentages
- **Tip**: Accept pivot suggestions when they make sense

#### Neglecting Curve
- **Problem**: Too many expensive or cheap cards
- **Solution**: Monitor the curve display constantly
- **Tip**: Prioritize curve over minor value differences

#### Panic Picking
- **Problem**: Making rushed decisions under time pressure
- **Solution**: Trust the AI's quick analysis
- **Tip**: The top recommendation is almost always correct

## Troubleshooting

### Common Issues

#### "No Recommendations Appearing"
**Possible Causes**:
- Hearthstone not detected
- Arena mode not active
- Cards not recognized

**Solutions**:
1. Restart both Hearthstone and Arena Bot
2. Verify Hearthstone is in windowed or borderless mode
3. Check that Arena draft is actually active
4. Update Arena Bot to latest version

#### "Low Confidence Warnings"
**Possible Causes**:
- API connectivity issues
- New cards not yet in database
- Unusual card combinations

**Solutions**:
1. Check internet connection
2. Wait for data to refresh (10-15 minutes)
3. Use manual override if needed
4. Trust fallback recommendations as baseline

#### "Recommendations Seem Wrong"
**Possible Causes**:
- Misread game state
- Personal playstyle differences
- Unusual meta shift

**Solutions**:
1. Check if archetype detection is correct
2. Verify hero class is properly detected
3. Consider if you're playing off-meta strategy
4. Use override system and track results

#### "Performance Issues"
**Possible Causes**:
- Insufficient RAM
- Background processes
- Outdated graphics drivers

**Solutions**:
1. Close unnecessary programs
2. Lower Hearthstone graphics settings
3. Restart both applications
4. Check system requirements

### Getting Help

#### In-App Support
- Press F1 for quick help overlay
- Click "?" next to any recommendation for details
- Use the Coach chat for specific questions

#### Community Resources
- Arena Bot Discord server: [Link]
- Reddit community: r/ArenaBot
- YouTube tutorials: [Channel Link]
- Twitch streamers using Arena Bot: [List]

#### Reporting Issues
When reporting problems, include:
1. Arena Bot version number
2. Hearthstone version
3. Operating system
4. Screenshot of the issue
5. Steps to reproduce

### Advanced Configuration

#### Settings Menu Access
Right-click on the Arena Bot system tray icon and select "Settings"

#### Key Configuration Options

**Performance Settings**:
- Recommendation delay (0-3 seconds)
- Visual overlay opacity (0-100%)
- Memory usage limit (2-8 GB)

**Data Settings**:
- Auto-update frequency (1-24 hours)
- Offline mode preferences
- Personal data tracking (on/off)

**Display Settings**:
- Compact vs. detailed view
- Color themes (light/dark/high contrast)
- Font size adjustment

**Privacy Settings**:
- Anonymous usage statistics
- Personal performance tracking
- Draft sharing preferences

## Conclusion

The Arena Bot AI v2 represents the pinnacle of Hearthstone Arena assistance. By understanding its recommendations, trusting its statistical analysis, and learning from its coaching, you'll see dramatic improvements in your Arena performance.

Remember: The AI is your coach, not your replacement. The goal is to help you become a better Arena player by understanding the reasoning behind optimal plays. Over time, you'll internalize these lessons and develop expert-level intuition for Arena drafting.

Good luck, and may your Arena runs be legendary! 🏆

---

*For technical support, advanced configuration, or feature requests, please refer to the technical documentation or contact our support team.*