"""
Compare two models to see what changed
"""
import torch
from pathlib import Path
import glob

# Find all models
model_files = glob.glob('saved_models/*/best_model.pth')
model_files.sort(key=lambda x: Path(x).stat().st_mtime)

if len(model_files) < 2:
    print("❌ Need at least 2 models to compare!")
    exit(1)

# Get the two most recent
old_model = model_files[-2]
new_model = model_files[-1]

print("="*60)
print("🔍 COMPARING MODELS")
print("="*60)

print(f"\n📁 OLD MODEL: {old_model}")
old_checkpoint = torch.load(old_model, map_location='cpu')
print(f"   Accuracy: {old_checkpoint.get('accuracy', 'N/A')}")
print(f"   Threshold: {old_checkpoint.get('threshold', 'N/A')}")
print(f"   Epoch: {old_checkpoint.get('epoch', 'N/A')}")
print(f"   Features: {old_checkpoint['config'].get('num_acoustic_features', 'N/A')}")

print(f"\n📁 NEW MODEL: {new_model}")
new_checkpoint = torch.load(new_model, map_location='cpu')
print(f"   Accuracy: {new_checkpoint.get('accuracy', 'N/A')}")
print(f"   Threshold: {new_checkpoint.get('threshold', 'N/A')}")
print(f"   Epoch: {new_checkpoint.get('epoch', 'N/A')}")
print(f"   Features: {new_checkpoint['config'].get('num_acoustic_features', 'N/A')}")

print("\n📊 DIFFERENCES:")
old_acc = old_checkpoint.get('accuracy', 0)
new_acc = new_checkpoint.get('accuracy', 0)
diff = new_acc - old_acc

print(f"   Accuracy change: {diff:+.2f}%")

if diff < 0:
    print(f"\n⚠️  NEW MODEL IS WORSE by {abs(diff):.2f}%!")
    print(f"\n💡 RECOMMENDATION: Use old model at:")
    print(f"   {old_model}")
else:
    print(f"\n✅ New model is better by {diff:.2f}%")