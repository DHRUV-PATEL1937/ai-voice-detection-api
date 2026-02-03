"""
Quick fix for import errors in feature_extractor.py
"""
from pathlib import Path

feature_extractor_path = Path('ml_models/feature_extractor.py')

# Read the file
content = feature_extractor_path.read_text(encoding='utf-8')

# Fix imports
content = content.replace(
    'from emotional_features import EmotionalFeatureExtractor',
    'from .emotional_features import EmotionalFeatureExtractor'
)
content = content.replace(
    'from articulatory_features import ArticulatoryFeatureExtractor',
    'from .articulatory_features import ArticulatoryFeatureExtractor'
)
content = content.replace(
    'from speaking_style_features import SpeakingStyleExtractor',
    'from .speaking_style_features import SpeakingStyleExtractor'
)

# Write back
feature_extractor_path.write_text(content, encoding='utf-8')

print("✅ Fixed imports in feature_extractor.py")