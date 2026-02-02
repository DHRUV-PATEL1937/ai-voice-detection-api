"""
Diagnose why humans are being detected as AI
"""
import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.dataset import VoiceDataset
from ml_models.human_voice_authenticator import HumanVoiceAuthenticator

def diagnose_model(model_path):
    """Analyze model behavior on human and AI samples"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    num_features = checkpoint['config']['num_acoustic_features']
    
    model = HumanVoiceAuthenticator(num_features, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    human_centroid = checkpoint['human_centroid'].to(device)
    current_threshold = checkpoint.get('threshold', 0.5)
    
    print("="*60)
    print("🔍 MODEL DIAGNOSTICS")
    print("="*60)
    print(f"\nCurrent threshold: {current_threshold:.4f}")
    
    # Load validation data
    val_dataset = VoiceDataset('dataset', split='validation', cache_features=True)
    
    human_distances = []
    ai_distances = []
    
    human_samples = []
    ai_samples = []
    
    print("\n📊 Computing distances...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset))):
            mel_spec, acoustic_feats, label = val_dataset[idx]
            
            mel_spec = mel_spec.unsqueeze(0).to(device)
            acoustic_feats = acoustic_feats.unsqueeze(0).to(device)
            label = label.item()
            
            embedding = model(mel_spec, acoustic_feats)
            distance = torch.norm(embedding - human_centroid, dim=1).item()
            
            if label == 1:  # Human
                human_distances.append(distance)
                human_samples.append(idx)
            else:  # AI
                ai_distances.append(distance)
                ai_samples.append(idx)
    
    human_distances = np.array(human_distances)
    ai_distances = np.array(ai_distances)
    
    # Statistics
    print("\n" + "="*60)
    print("📊 DISTANCE STATISTICS")
    print("="*60)
    
    print(f"\nHUMAN Voices ({len(human_distances)} samples):")
    print(f"  Mean:     {np.mean(human_distances):.4f}")
    print(f"  Std Dev:  {np.std(human_distances):.4f}")
    print(f"  Min:      {np.min(human_distances):.4f}")
    print(f"  Max:      {np.max(human_distances):.4f}")
    print(f"  Median:   {np.median(human_distances):.4f}")
    print(f"  25th %:   {np.percentile(human_distances, 25):.4f}")
    print(f"  75th %:   {np.percentile(human_distances, 75):.4f}")
    print(f"  95th %:   {np.percentile(human_distances, 95):.4f}")
    
    if len(ai_distances) > 0:
        print(f"\nAI Voices ({len(ai_distances)} samples):")
        print(f"  Mean:     {np.mean(ai_distances):.4f}")
        print(f"  Std Dev:  {np.std(ai_distances):.4f}")
        print(f"  Min:      {np.min(ai_distances):.4f}")
        print(f"  Max:      {np.max(ai_distances):.4f}")
        print(f"  Median:   {np.median(ai_distances):.4f}")
    
    # Analyze with current threshold
    print("\n" + "="*60)
    print("🎯 CURRENT THRESHOLD ANALYSIS")
    print("="*60)
    
    human_correct = np.sum(human_distances < current_threshold)
    human_wrong = np.sum(human_distances >= current_threshold)
    human_acc = 100 * human_correct / len(human_distances)
    
    print(f"\nCurrent Threshold: {current_threshold:.4f}")
    print(f"\nHuman Detection:")
    print(f"  Correct (< threshold): {human_correct} ({human_acc:.2f}%)")
    print(f"  Wrong (>= threshold):  {human_wrong} ({100-human_acc:.2f}%) ⚠️")
    
    if len(ai_distances) > 0:
        ai_correct = np.sum(ai_distances >= current_threshold)
        ai_wrong = np.sum(ai_distances < current_threshold)
        ai_acc = 100 * ai_correct / len(ai_distances)
        
        print(f"\nAI Detection:")
        print(f"  Correct (>= threshold): {ai_correct} ({ai_acc:.2f}%)")
        print(f"  Wrong (< threshold):    {ai_wrong} ({100-ai_acc:.2f}%)")
    
    # Suggest better thresholds
    print("\n" + "="*60)
    print("💡 RECOMMENDED THRESHOLDS")
    print("="*60)
    
    # Option 1: 95th percentile of human (very conservative)
    threshold_95 = np.percentile(human_distances, 95)
    human_acc_95 = 100 * np.sum(human_distances < threshold_95) / len(human_distances)
    
    print(f"\nOption 1: 95th percentile (Conservative)")
    print(f"  Threshold: {threshold_95:.4f}")
    print(f"  Human Accuracy: {human_acc_95:.2f}%")
    
    if len(ai_distances) > 0:
        ai_acc_95 = 100 * np.sum(ai_distances >= threshold_95) / len(ai_distances)
        balanced_95 = (human_acc_95 + ai_acc_95) / 2
        print(f"  AI Detection: {ai_acc_95:.2f}%")
        print(f"  Balanced: {balanced_95:.2f}%")
    
    # Option 2: Mean + 2*std (statistical)
    threshold_2std = np.mean(human_distances) + 2 * np.std(human_distances)
    human_acc_2std = 100 * np.sum(human_distances < threshold_2std) / len(human_distances)
    
    print(f"\nOption 2: Mean + 2σ (Statistical)")
    print(f"  Threshold: {threshold_2std:.4f}")
    print(f"  Human Accuracy: {human_acc_2std:.2f}%")
    
    if len(ai_distances) > 0:
        ai_acc_2std = 100 * np.sum(ai_distances >= threshold_2std) / len(ai_distances)
        balanced_2std = (human_acc_2std + ai_acc_2std) / 2
        print(f"  AI Detection: {ai_acc_2std:.2f}%")
        print(f"  Balanced: {balanced_2std:.2f}%")
    
    # Option 3: Optimal (if AI samples available)
    if len(ai_distances) > 0:
        # Find threshold that maximizes balanced accuracy
        thresholds = np.linspace(
            min(np.min(human_distances), np.min(ai_distances)),
            max(np.max(human_distances), np.max(ai_distances)),
            100
        )
        
        best_balanced = 0
        best_threshold = current_threshold
        
        for t in thresholds:
            h_acc = 100 * np.sum(human_distances < t) / len(human_distances)
            a_acc = 100 * np.sum(ai_distances >= t) / len(ai_distances)
            balanced = (h_acc + a_acc) / 2
            
            if balanced > best_balanced:
                best_balanced = balanced
                best_threshold = t
        
        print(f"\nOption 3: Optimal Balanced (Data-driven)")
        print(f"  Threshold: {best_threshold:.4f}")
        
        human_acc_opt = 100 * np.sum(human_distances < best_threshold) / len(human_distances)
        ai_acc_opt = 100 * np.sum(ai_distances >= best_threshold) / len(ai_distances)
        
        print(f"  Human Accuracy: {human_acc_opt:.2f}%")
        print(f"  AI Detection: {ai_acc_opt:.2f}%")
        print(f"  Balanced: {best_balanced:.2f}%")
    
    # Visualize distribution
    print("\n" + "="*60)
    print("📊 Generating distribution plot...")
    print("="*60)
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(human_distances, bins=50, alpha=0.7, label='Human', color='blue', density=True)
    if len(ai_distances) > 0:
        plt.hist(ai_distances, bins=50, alpha=0.7, label='AI', color='red', density=True)
    
    plt.axvline(current_threshold, color='black', linestyle='--', linewidth=2, label=f'Current Threshold ({current_threshold:.3f})')
    plt.axvline(threshold_95, color='green', linestyle='--', linewidth=2, label=f'95th Percentile ({threshold_95:.3f})')
    
    plt.xlabel('Distance from Human Centroid')
    plt.ylabel('Density')
    plt.title('Distribution of Distances from Human Centroid')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = Path('diagnostics_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {plot_path}")
    
    # Find which human samples are misclassified
    if human_wrong > 0:
        print("\n" + "="*60)
        print("⚠️  MISCLASSIFIED HUMAN SAMPLES")
        print("="*60)
        
        misclassified_indices = [human_samples[i] for i in range(len(human_distances)) 
                                if human_distances[i] >= current_threshold]
        
        print(f"\nFound {len(misclassified_indices)} human samples detected as AI:")
        for idx in misclassified_indices[:10]:  # Show first 10
            distance = human_distances[human_samples.index(idx)]
            print(f"  Sample {idx}: distance = {distance:.4f}")
    
    print("\n" + "="*60)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*60)
    
    return {
        'current_threshold': current_threshold,
        'recommended_95': threshold_95,
        'recommended_2std': threshold_2std,
        'human_mean': np.mean(human_distances),
        'human_std': np.std(human_distances)
    }


if __name__ == "__main__":
    import glob
    
    # Find the latest model
    model_files = glob.glob('saved_models/human_authenticator_*/best_model.pth')
    
    if not model_files:
        print("❌ No trained model found!")
        print("Please train the model first: python train_human_authenticator.py")
        exit(1)
    
    latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
    
    print(f"📂 Using model: {latest_model}\n")
    
    results = diagnose_model(latest_model)