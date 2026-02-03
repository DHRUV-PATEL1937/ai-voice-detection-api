"""
Update the threshold in saved model
"""
import torch
from pathlib import Path
import glob

def update_threshold(model_path, new_threshold):
    """Update threshold in model checkpoint"""
    
    checkpoint = torch.load(model_path)
    old_threshold = checkpoint.get('threshold', 0.5)
    
    print(f"Old threshold: {old_threshold:.4f}")
    print(f"New threshold: {new_threshold:.4f}")
    
    checkpoint['threshold'] = new_threshold
    
    torch.save(checkpoint, model_path)
    
    print(f"✅ Updated threshold in {model_path}")

if __name__ == "__main__":
    # Find latest model
    model_files = glob.glob('saved_models/human_authenticator_20260201_144346/best_model.pth')
    latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
    
    # Get recommended threshold from user
    print(f"Current model: {latest_model}\n")
    print("Run diagnose_detector.py first to see recommended thresholds!")
    
    new_threshold = float(input("\nEnter new threshold (e.g., 0.85): "))
    
    update_threshold(latest_model, new_threshold)
    
    # Also update latest model
    latest_path = Path(latest_model).parent / 'latest_model.pth'
    if latest_path.exists():
        update_threshold(str(latest_path), new_threshold)