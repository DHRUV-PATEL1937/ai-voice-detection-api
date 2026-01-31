"""
Dataset Inspector - Check your dataset quality
"""
import os
from pathlib import Path
from collections import defaultdict
import librosa

def inspect_dataset(base_path='dataset'):
    """Inspect and validate dataset"""
    
    stats = defaultdict(lambda: defaultdict(int))
    issues = []
    
    base = Path(base_path)
    
    print("=" * 60)
    print("📊 DATASET INSPECTION REPORT")
    print("=" * 60)
    
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} SET:")
        print("-" * 60)
        
        for category in ['human', 'ai_generated']:
            for language in ['tamil', 'english', 'hindi', 'malayalam', 'telugu']:
                path = base / split / category / language
                
                if not path.exists():
                    issues.append(f"Missing directory: {path}")
                    continue
                
                # Count files
                audio_files = list(path.glob('*.mp3')) + list(path.glob('*.wav'))
                count = len(audio_files)
                stats[split][f'{category}_{language}'] = count
                
                print(f"  {category:15} | {language:10} | {count:4} files")
                
                # Check audio quality (sample first file)
                if audio_files:
                    try:
                        y, sr = librosa.load(audio_files[0], sr=None, duration=1)
                        duration = librosa.get_duration(y=y, sr=sr)
                        # We'll check more later
                    except Exception as e:
                        issues.append(f"Error loading {audio_files[0]}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    
    for split in ['train', 'validation', 'test']:
        total = sum(stats[split].values())
        print(f"{split.upper():12}: {total:4} total files")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No issues found!")
    
    return stats, issues

if __name__ == "__main__":
    inspect_dataset()