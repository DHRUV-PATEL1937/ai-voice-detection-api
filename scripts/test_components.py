"""
Test all components before training
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.audio_processor import AudioProcessor
from ml_models.feature_extractor import FeatureExtractor

def test_components():
    """Test audio processing and feature extraction"""
    
    print("=" * 60)
    print("🧪 TESTING COMPONENTS")
    print("=" * 60)
    
    # Find test files
    base_path = Path("dataset/train")
    
    test_files = []
    for category in ['human', 'ai_generated']:
        for language in ['tamil', 'english', 'hindi', 'malayalam', 'telugu']:
            path = base_path / category / language
            files = list(path.glob("*.mp3")) + list(path.glob("*.wav"))
            if files:
                test_files.append((files[0], category, language))
                break
        if test_files:
            break
    
    if not test_files:
        print("❌ No audio files found in dataset!")
        print("Please check your dataset structure:")
        print("  dataset/train/human/{language}/*.mp3")
        print("  dataset/train/ai_generated/{language}/*.mp3")
        return False
    
    # Test each component
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    for test_file, category, language in test_files:
        print(f"\n📝 Testing: {category}/{language}/{test_file.name}")
        print("-" * 60)
        
        try:
            # Test 1: Load audio
            print("  [1/3] Loading audio...", end=" ")
            audio, sr = processor.load_audio_file(str(test_file))
            print(f"✅ Shape: {audio.shape}, SR: {sr}")
            
            # Test 2: Preprocess
            print("  [2/3] Preprocessing...", end=" ")
            audio_processed = processor.preprocess(audio, sr)
            print(f"✅ Shape: {audio_processed.shape}")
            
            # Test 3: Extract features
            print("  [3/3] Extracting features...", end=" ")
            mel_spec, acoustic_features = extractor.extract_all_features(audio_processed)
            print(f"✅ Mel: {mel_spec.shape}, Acoustic: {len(acoustic_features)}")
            
            print("\n  🎯 Sample acoustic features:")
            for i, (key, value) in enumerate(list(acoustic_features.items())[:5]):
                print(f"     - {key}: {value:.4f}")
            
        except Exception as e:
            print(f"❌ FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_components()
    sys.exit(0 if success else 1)