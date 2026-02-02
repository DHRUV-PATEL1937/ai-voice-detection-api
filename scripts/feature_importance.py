"""
Analyze which features are most discriminative
"""
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.dataset import VoiceDataset
from ml_models.human_voice_authenticator import HumanVoiceAuthenticator

def analyze_features(model_path):
    """See which features differ most between human and AI"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load data
    val_dataset = VoiceDataset('dataset', split='validation', cache_features=True)
    
    # Collect features
    human_features = []
    ai_features = []
    
    print("Collecting features...")
    for idx in tqdm(range(len(val_dataset))):
        mel_spec, acoustic_feats, label = val_dataset[idx]
        
        if label.item() == 1:  # Human
            human_features.append(acoustic_feats.numpy())
        else:  # AI
            ai_features.append(acoustic_feats.numpy())
    
    human_features = np.array(human_features)
    ai_features = np.array(ai_features)
    
    # Get feature names from cached features
    if hasattr(val_dataset, 'cached_features') and val_dataset.cached_features[0]:
        feature_names = sorted(val_dataset.cached_features[0]['acoustic_features'].keys())
    else:
        feature_names = [f'feature_{i}' for i in range(human_features.shape[1])]
    
    # Compute mean difference
    human_mean = np.mean(human_features, axis=0)
    ai_mean = np.mean(ai_features, axis=0) if len(ai_features) > 0 else np.zeros_like(human_mean)
    
    difference = np.abs(human_mean - ai_mean)
    
    # Sort by difference
    indices = np.argsort(difference)[::-1]
    
    print("\n" + "="*60)
    print("TOP 20 MOST DISCRIMINATIVE FEATURES")
    print("="*60)
    
    for i, idx in enumerate(indices[:20]):
        if idx < len(feature_names):
            print(f"{i+1:2d}. {feature_names[idx]:40s} | Δ = {difference[idx]:.4f}")
    
    return feature_names, difference

if __name__ == "__main__":
    import glob
    
    model_files = glob.glob('saved_models/human_authenticator_*/best_model.pth')
    latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
    
    analyze_features(latest_model)