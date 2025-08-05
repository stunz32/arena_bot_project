# AI Models Directory

This directory contains machine learning models for the Arena Bot AI system.

## Model Files

The following model files are expected but not required:
- `base_value_model.lgb` - Base card value prediction model
- `tempo_model.lgb` - Tempo score prediction model  
- `value_model.lgb` - Value score prediction model
- `synergy_model.lgb` - Synergy score prediction model

## Fallback Behavior

If model files are missing, the system will automatically fall back to heuristic-based evaluation with the following behavior:
- INFO log messages instead of WARNING messages
- Models marked as loaded in fallback mode (None value)
- Full functionality maintained using proven heuristic algorithms

## Model Creation

Models should be trained using LightGBM and saved in .lgb format. The system expects models to:
- Accept feature vectors extracted from card and deck state
- Return predictions in the range 0-100
- Be thread-safe for concurrent access