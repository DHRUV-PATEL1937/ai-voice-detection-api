"""
Feature Extraction for Voice Detection
Extracts: Mel Spectrograms, MFCCs, Acoustic Features
"""
import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """Extract features from audio for AI detection"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_mel_spectrogram(self, audio):
        """
        Extract Mel Spectrogram
        This captures frequency patterns over time
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=128,       # Number of mel bands
            n_fft=2048,       # FFT window size
            hop_length=512,   # Hop length
            fmax=8000         # Maximum frequency
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
    
    def extract_mfcc(self, audio):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=13
        )
        return mfcc
    
    def extract_pitch(self, audio):
        """
        Extract pitch (F0) features
        AI voices often have unnatural pitch consistency
        """
        # Use YIN algorithm for pitch detection
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=self.sr
        )
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_range': 0,
                'pitch_variance': 0
            }
        
        return {
            'pitch_mean': np.mean(f0_clean),
            'pitch_std': np.std(f0_clean),
            'pitch_range': np.max(f0_clean) - np.min(f0_clean),
            'pitch_variance': np.var(f0_clean)
        }
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_contrast_std': np.std(spectral_contrast),
        }
    
    def extract_energy_features(self, audio):
        """
        Extract energy-related features
        AI voices may have unnatural energy stability
        """
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        return {
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms),
            'energy_max': np.max(rms),
            'energy_min': np.min(rms),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
        }
    
    def extract_temporal_features(self, audio):
        """
        Extract temporal features (pauses, speech rate, etc.)
        Humans have natural disfluencies
        """
        # Detect speech/silence intervals
        intervals = librosa.effects.split(audio, top_db=20)
        
        # Calculate pause statistics
        silence_durations = []
        for i in range(len(intervals) - 1):
            silence_duration = (intervals[i+1][0] - intervals[i][1]) / self.sr
            silence_durations.append(silence_duration)
        
        # Energy-based features
        rms = librosa.feature.rms(y=audio)[0]
        energy_diff = np.diff(rms)
        
        return {
            'num_pauses': len(silence_durations),
            'avg_pause_duration': np.mean(silence_durations) if silence_durations else 0,
            'max_pause_duration': np.max(silence_durations) if silence_durations else 0,
            'energy_sudden_changes': np.sum(np.abs(energy_diff) > 0.1),
            'speech_segments': len(intervals),
        }
    
    def extract_all_acoustic_features(self, audio):
        """
        Extract all acoustic features and return as a flat dictionary
        """
        features = {}
        
        # MFCC statistics
        mfcc = self.extract_mfcc(audio)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # Pitch features
        pitch_features = self.extract_pitch(audio)
        for key, value in pitch_features.items():
            features[key] = value
        
        # Spectral features
        spectral_features = self.extract_spectral_features(audio)
        for key, value in spectral_features.items():
            features[key] = value
        
        # Energy features
        energy_features = self.extract_energy_features(audio)
        for key, value in energy_features.items():
            features[key] = value
        
        # Temporal features
        temporal_features = self.extract_temporal_features(audio)
        for key, value in temporal_features.items():
            features[key] = value
        
        # Flatten MFCC mean and std
        flattened_features = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    flattened_features[f'{key}_{i}'] = float(v)  # Ensure float
            else:
                flattened_features[key] = float(value)  # Ensure float
        
        # ENSURE CONSISTENT FEATURE COUNT
        # Sort keys for consistency
        sorted_keys = sorted(flattened_features.keys())
        
        # Pad to 50 features if needed
        target_count = 50
        while len(sorted_keys) < target_count:
            sorted_keys.append(f'padding_{len(sorted_keys)}')
            flattened_features[f'padding_{len(sorted_keys)-1}'] = 0.0
        
        return flattened_features
    
    def extract_all_features(self, audio):
        """
        Extract both mel spectrogram and acoustic features
        Returns: (mel_spec, acoustic_features_dict)
        """
        mel_spec = self.extract_mel_spectrogram(audio)
        acoustic_features = self.extract_all_acoustic_features(audio)
        
        return mel_spec, acoustic_features


# Test the extractor
if __name__ == "__main__":
    try:
        from ml_models.audio_processor import AudioProcessor
    except ImportError:
        from audio_processor import AudioProcessor
    
    from pathlib import Path
    
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    # Find any audio file from your dataset
    dataset_path = Path("../dataset/train/human/english")
    
    # Get first available audio file
    audio_files = list(dataset_path.glob("*.mp3")) + list(dataset_path.glob("*.wav"))
    
    if not audio_files:
        print("❌ No audio files found in dataset/train/human/english/")
        print("ℹ️  Trying other languages...")
        
        # Try other languages
        for lang in ['tamil', 'hindi', 'malayalam', 'telugu']:
            dataset_path = Path(f"../dataset/train/human/{lang}")
            audio_files = list(dataset_path.glob("*.mp3")) + list(dataset_path.glob("*.wav"))
            if audio_files:
                print(f"✅ Found files in {lang}")
                break
        
        if not audio_files:
            print("❌ No audio files found in any language!")
            print("Please check your dataset structure")
            exit(1)
    
    test_file = audio_files[0]
    print(f"🎵 Testing with: {test_file}")
    
    try:
        # Process audio
        audio = processor.process_from_file(str(test_file))
        print(f"✅ Audio processed: shape={audio.shape}")
        
        # Extract features
        mel_spec, acoustic_features = extractor.extract_all_features(audio)
        
        print(f"✅ Mel Spectrogram shape: {mel_spec.shape}")
        print(f"✅ Number of acoustic features: {len(acoustic_features)}")
        print(f"\n🔍 First 5 acoustic features:")
        for i, (key, value) in enumerate(list(acoustic_features.items())[:5]):
            print(f"  {i+1}. {key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()