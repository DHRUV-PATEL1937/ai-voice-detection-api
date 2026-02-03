"""
Delete all old gTTS AI samples - they're useless!
We'll replace with REAL professional AI voices
"""
from pathlib import Path
import shutil

def clean_old_ai_samples():
    """Remove all current AI-generated samples"""
    
    print("🗑️  CLEANING OLD AI SAMPLES")
    print("="*60)
    
    total_deleted = 0
    
    for split in ['train', 'validation', 'test']:
        for language in ['tamil', 'english', 'hindi', 'malayalam', 'telugu']:
            ai_dir = Path(f'dataset/{split}/ai_generated/{language}')
            
            if ai_dir.exists():
                files = list(ai_dir.glob('*.mp3')) + list(ai_dir.glob('*.wav'))
                count = len(files)
                
                if count > 0:
                    print(f"📂 {split}/{language}: Deleting {count} files...")
                    
                    for file in files:
                        file.unlink()
                    
                    total_deleted += count
    
    print("="*60)
    print(f"✅ Deleted {total_deleted} old AI samples")
    print("   Ready for professional AI voice generation!")
    print("="*60)

if __name__ == "__main__":
    response = input("⚠️  This will DELETE all AI samples. Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        clean_old_ai_samples()
        print("\n✅ Done! Now run: python scripts/collect_real_ai_voices.py")
    else:
        print("Cancelled.")