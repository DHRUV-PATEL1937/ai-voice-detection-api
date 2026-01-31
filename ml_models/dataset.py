"""
PyTorch Dataset for Voice Detection
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# Fix imports to work from project root
try:
    from ml_models.audio_processor import AudioProcessor
    from ml_models.feature_extractor import FeatureExtractor
except ImportError:
    from audio_processor import AudioProcessor
    from feature_extractor import FeatureExtractor

class VoiceDataset(Dataset):
    """Dataset for AI vs Human voice classification"""
    
    def __init__(self, data_dir, split='train', cache_features=True):
        """
        Args:
            data_dir: Base directory containing dataset
            split: 'train', 'validation', or 'test'
            cache_features: If True, cache extracted features to disk
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_dir = self.data_dir / 'cache' / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = AudioProcessor()
        self.extractor = FeatureExtractor()
        
        # Collect all audio files
        self.audio_files = []
        self.labels = []
        
        self._load_dataset()
        
        # Cache features if requested
        if cache_features:
            self._cache_features()
    
    def get_num_acoustic_features(self):
        """Get number of acoustic features"""
        # Return fixed number to ensure consistency
        return 50  # Fixed feature count
    
    def _load_dataset(self):
        """Load all audio file paths and labels"""
        
        print(f"📂 Loading {self.split} dataset...")
        
        split_dir = self.data_dir / self.split
        
        # Label mapping: AI_GENERATED = 0, HUMAN = 1
        categories = {
            'ai_generated': 0,
            'human': 1
        }
        
        languages = ['tamil', 'english', 'hindi', 'malayalam', 'telugu']
        
        for category, label in categories.items():
            for language in languages:
                lang_dir = split_dir / category / language
                
                if not lang_dir.exists():
                    continue
                
                # Get all audio files
                audio_files = list(lang_dir.glob('*.mp3')) + list(lang_dir.glob('*.wav'))
                
                for audio_file in audio_files:
                    self.audio_files.append(audio_file)
                    self.labels.append(label)
        
        print(f"  ✅ Loaded {len(self.audio_files)} audio files")
        print(f"  ✅ AI: {sum(1 for l in self.labels if l == 0)}, Human: {sum(1 for l in self.labels if l == 1)}")
    
    def _cache_features(self):
        """Pre-extract and cache all features"""
        
        cache_file = self.cache_dir / 'features.pkl'
        
        if cache_file.exists():
            print(f"  ℹ️  Loading cached features from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.cached_features = pickle.load(f)
            return
        
        print(f"  🔄 Extracting and caching features...")
        self.cached_features = []
        
        for audio_file in tqdm(self.audio_files, desc="Extracting features"):
            try:
                # Process audio
                audio = self.processor.process_from_file(str(audio_file))
                
                # Extract features
                mel_spec, acoustic_features = self.extractor.extract_all_features(audio)
                
                self.cached_features.append({
                    'mel_spec': mel_spec,
                    'acoustic_features': acoustic_features
                })
                
            except Exception as e:
                print(f"  ⚠️  Error processing {audio_file.name}: {e}")
                # Add dummy features
                self.cached_features.append(None)
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cached_features, f)
        
        print(f"  ✅ Features cached to {cache_file}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        Returns: (mel_spec, acoustic_features, label)
        """
        
        # Use cached features if available
        if hasattr(self, 'cached_features') and self.cached_features[idx] is not None:
            features = self.cached_features[idx]
            mel_spec = features['mel_spec']
            acoustic_features = features['acoustic_features']
        else:
            # Extract features on-the-fly
            try:
                audio = self.processor.process_from_file(str(self.audio_files[idx]))
                mel_spec, acoustic_features = self.extractor.extract_all_features(audio)
            except Exception as e:
                # Return zeros if error
                mel_spec = np.zeros((128, 157))
                acoustic_features = {f'feature_{i}': 0.0 for i in range(50)}
        
        label = self.labels[idx]
        
        # Convert to tensors
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dimension
        
        # Convert acoustic features dict to tensor (maintain consistent order)
        # Sort keys to ensure consistent ordering
        sorted_keys = sorted(acoustic_features.keys())
        acoustic_values = [acoustic_features[key] for key in sorted_keys]
        
        # PAD OR TRIM to ensure consistent length
        target_length = 50  # Set a fixed target length
        if len(acoustic_values) < target_length:
            # Pad with zeros
            acoustic_values += [0.0] * (target_length - len(acoustic_values))
        elif len(acoustic_values) > target_length:
            # Trim
            acoustic_values = acoustic_values[:target_length]
        
        acoustic_tensor = torch.FloatTensor(acoustic_values)
        label_tensor = torch.LongTensor([label])
        
        return mel_spec_tensor, acoustic_tensor, label_tensor

# Test the dataset
if __name__ == "__main__":
    # Test loading
    dataset = VoiceDataset('../dataset', split='train', cache_features=True)
    
    print(f"\n📊 Dataset Info:")
    print(f"  Total samples: {len(dataset)}")
    
    # Get one sample
    mel_spec, acoustic_feats, label = dataset[0]
    
    print(f"\n🔍 Sample Data:")
    print(f"  Mel Spectrogram shape: {mel_spec.shape}")
    print(f"  Acoustic features shape: {acoustic_feats.shape}")
    print(f"  Label: {'AI_GENERATED' if label.item() == 0 else 'HUMAN'}")