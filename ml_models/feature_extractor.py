"""
Advanced Feature Extraction for AI Voice Detection
Focuses on Human Authentication - detecting micro-signatures AI cannot replicate
"""
import librosa
import numpy as np
import warnings
from scipy import signal, stats
from scipy.fft import fft
from .emotional_features import EmotionalFeatureExtractor
from .articulatory_features import ArticulatoryFeatureExtractor
from .speaking_style_features import SpeakingStyleExtractor

warnings.filterwarnings('ignore')

class FeatureExtractor:
    """Extract features that authenticate human voices"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        self.emotional = EmotionalFeatureExtractor(sr=sr)
        self.articulatory = ArticulatoryFeatureExtractor(sr=sr)
        self.speaking_style = SpeakingStyleExtractor(sr=sr)
    
    def extract_mel_spectrogram(self, audio):
        """Extract Mel Spectrogram for CNN"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
    
    # ========================================================================
    # HUMAN AUTHENTICATION FEATURES - These detect real human characteristics
    # ========================================================================
    
    def extract_micro_prosody_features(self, audio):
        """
        Micro-prosodic variations - Humans have unconscious pitch/energy variations
        AI is too stable at millisecond level
        """
        features = {}
        
        # Extract fundamental frequency (pitch)
        f0 = librosa.yin(audio, fmin=65, fmax=400, sr=self.sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 10:
            # 1. Micro-jitter: Frame-to-frame pitch variation
            # Humans: 0.5-2.0, AI: 0.1-0.5
            frame_diffs = np.abs(np.diff(f0_clean))
            features['micro_jitter'] = np.mean(frame_diffs)
            features['jitter_variance'] = np.var(frame_diffs)
            features['jitter_max'] = np.max(frame_diffs)
            
            # 2. Pitch contour complexity (curvature)
            # Humans have complex natural curves, AI is too smooth
            if len(f0_clean) > 20:
                pitch_velocity = np.diff(f0_clean)
                pitch_acceleration = np.diff(pitch_velocity)
                features['pitch_complexity'] = np.std(pitch_acceleration)
                features['pitch_curvature'] = np.mean(np.abs(pitch_acceleration))
            else:
                features['pitch_complexity'] = 0
                features['pitch_curvature'] = 0
            
            # 3. Long-term vs short-term stability
            # AI is stable at all time scales, humans vary more at short scales
            window_size = max(len(f0_clean) // 10, 5)
            short_term_vars = []
            for i in range(0, len(f0_clean) - window_size, window_size):
                window = f0_clean[i:i+window_size]
                short_term_vars.append(np.var(window))
            
            features['pitch_stability_ratio'] = np.var(short_term_vars) / (np.var(f0_clean) + 1e-8)
            
        else:
            features['micro_jitter'] = 0
            features['jitter_variance'] = 0
            features['jitter_max'] = 0
            features['pitch_complexity'] = 0
            features['pitch_curvature'] = 0
            features['pitch_stability_ratio'] = 0
        
        # 4. Shimmer: Amplitude variation (energy jitter)
        rms = librosa.feature.rms(y=audio)[0]
        shimmer = np.abs(np.diff(rms)) / (np.mean(rms) + 1e-8)
        features['micro_shimmer'] = np.mean(shimmer)
        features['shimmer_variance'] = np.var(shimmer)
        
        return features
    
    def extract_breathing_signatures(self, audio):
        """
        Natural breathing patterns - AI doesn't breathe!
        Humans have audible breath sounds and natural pauses
        """
        features = {}
        
        rms = librosa.feature.rms(y=audio)[0]
        
        # 1. Breath detection (low energy regions with specific spectral profile)
        threshold = np.mean(rms) * 0.15
        low_energy_regions = rms < threshold
        
        # Count breath-like events
        breath_starts = np.where(np.diff(low_energy_regions.astype(int)) == 1)[0]
        features['breath_count'] = len(breath_starts)
        
        # 2. Breath regularity (humans breathe somewhat regularly)
        if len(breath_starts) > 2:
            breath_intervals = np.diff(breath_starts)
            features['breath_regularity'] = np.std(breath_intervals) / (np.mean(breath_intervals) + 1e-8)
        else:
            features['breath_regularity'] = 0
        
        # 3. Pause characteristics (natural vs synthetic)
        # Natural pauses have gradual energy decay, synthetic are abrupt
        silent_regions = rms < (np.mean(rms) * 0.05)
        
        if np.sum(silent_regions) > 0:
            # Find pause boundaries
            pause_changes = np.diff(silent_regions.astype(int))
            pause_starts = np.where(pause_changes == 1)[0]
            pause_ends = np.where(pause_changes == -1)[0]
            
            # Measure transition gradients (how quickly energy drops)
            gradients = []
            for start in pause_starts[:10]:  # Check first 10 pauses
                if start > 5:
                    gradient = np.mean(np.abs(np.diff(rms[start-5:start])))
                    gradients.append(gradient)
            
            features['pause_gradient'] = np.mean(gradients) if gradients else 0
        else:
            features['pause_gradient'] = 0
        
        features['natural_pause_ratio'] = np.sum(silent_regions) / len(rms)
        
        return features
    
    def extract_spectral_irregularities(self, audio):
        """
        Spectral micro-irregularities - Humans have natural noise and roughness
        AI synthesis is too spectrally clean
        """
        features = {}
        
        # Compute STFT
        D = librosa.stft(audio, n_fft=2048)
        magnitude = np.abs(D)
        
        # 1. High-frequency roughness (breathiness)
        # Humans have natural high-frequency noise from breath turbulence
        high_freq_start = magnitude.shape[0] // 2
        high_freq_energy = np.sum(magnitude[high_freq_start:, :], axis=0)
        features['spectral_roughness'] = np.std(high_freq_energy)
        features['hf_energy_ratio'] = np.mean(high_freq_energy) / (np.mean(magnitude) + 1e-8)
        
        # 2. Harmonic-to-noise ratio (HNR)
        # AI is too clean (high HNR), humans have natural noise (lower HNR)
        harmonics, noise = librosa.effects.hpss(audio)
        harmonic_energy = np.sum(harmonics**2)
        noise_energy = np.sum(noise**2)
        features['hnr'] = 10 * np.log10((harmonic_energy / (noise_energy + 1e-8)) + 1e-8)
        features['noise_ratio'] = noise_energy / (harmonic_energy + noise_energy + 1e-8)
        
        # 3. Spectral entropy (randomness)
        # Humans have higher entropy due to natural variations
        spectral_entropy_values = []
        for frame in magnitude.T:
            prob = frame / (np.sum(frame) + 1e-8)
            entropy = -np.sum(prob * np.log2(prob + 1e-8))
            spectral_entropy_values.append(entropy)
        
        features['spectral_entropy_mean'] = np.mean(spectral_entropy_values)
        features['spectral_entropy_std'] = np.std(spectral_entropy_values)
        
        # 4. Spectral flux (frame-to-frame spectral change)
        # Humans have more micro-variations
        spectral_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        features['spectral_flux_std'] = np.std(spectral_flux)
        
        return features
    
    def extract_temporal_micro_patterns(self, audio):
        """
        Temporal micro-patterns - Humans have timing irregularities
        AI has too-perfect timing
        """
        features = {}
        
        # 1. Speech rate variability
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        peaks, _ = signal.find_peaks(onset_env, height=np.mean(onset_env))
        
        if len(peaks) > 3:
            # Inter-onset intervals (syllable timing)
            intervals = np.diff(peaks)
            
            # Coefficient of variation (humans are irregular)
            features['timing_irregularity'] = np.std(intervals) / (np.mean(intervals) + 1e-8)
            features['timing_cv'] = stats.variation(intervals) if len(intervals) > 0 else 0
            
            # Rhythm complexity (entropy of interval distribution)
            if len(intervals) > 5:
                hist, _ = np.histogram(intervals, bins=10)
                prob = hist / (np.sum(hist) + 1e-8)
                features['rhythm_entropy'] = -np.sum(prob * np.log2(prob + 1e-8))
            else:
                features['rhythm_entropy'] = 0
        else:
            features['timing_irregularity'] = 0
            features['timing_cv'] = 0
            features['rhythm_entropy'] = 0
        
        # 2. Energy micro-fluctuations
        rms = librosa.feature.rms(y=audio)[0]
        
        # Detrend to isolate micro-fluctuations
        detrended = signal.detrend(rms)
        
        # FFT of energy fluctuations
        fft_energy = np.abs(fft(detrended))
        
        # High-frequency energy fluctuations (nervousness, micro-adjustments)
        high_freq_start = len(fft_energy) // 4
        features['energy_micro_fluctuation'] = np.sum(fft_energy[high_freq_start:])
        
        # 3. Attack/decay characteristics
        # How quickly energy rises and falls (humans have organic curves)
        energy_diff = np.diff(rms)
        features['energy_attack_sharpness'] = np.percentile(energy_diff[energy_diff > 0], 95) if np.any(energy_diff > 0) else 0
        features['energy_decay_smoothness'] = np.percentile(np.abs(energy_diff[energy_diff < 0]), 5) if np.any(energy_diff < 0) else 0
        
        return features
    
    def extract_vocal_tract_signatures(self, audio):
        """
        Physical vocal tract artifacts - Real vocal tracts have unique characteristics
        AI synthesis models don't perfectly replicate physical resonances
        """
        features = {}
        
        # 1. Formant tracking (vocal tract resonances)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # First 3-4 MFCCs relate to formant structure
        for i in range(4):
            formant_track = mfccs[i, :]
            
            # Micro-variations in formants (humans have slight wobble)
            features[f'formant_{i}_micro_var'] = np.std(np.diff(formant_track))
            
            # Formant transition speed
            features[f'formant_{i}_transition'] = np.mean(np.abs(np.diff(formant_track)))
        
        # 2. Vocal fry detection (creaky voice)
        # Low-frequency irregularity, common in natural speech
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        low_zcr_ratio = np.sum(zcr < np.percentile(zcr, 25)) / len(zcr)
        features['vocal_fry_indicator'] = low_zcr_ratio
        
        # 3. Subharmonics (imperfect vocal fold vibration)
        # Humans sometimes have subharmonics, AI doesn't
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features['subharmonic_indicator'] = np.sum(spec_centroid < 150) / len(spec_centroid)
        
        return features
    
    def extract_all_human_auth_features(self, audio):
        """Combine all human authentication features"""
        all_features = {}
        
        # Original features
        prosody = self.extract_micro_prosody_features(audio)
        breathing = self.extract_breathing_signatures(audio)
        spectral = self.extract_spectral_irregularities(audio)
        temporal = self.extract_temporal_micro_patterns(audio)
        vocal = self.extract_vocal_tract_signatures(audio)
        
        # NEW ENHANCED FEATURES
        emotional = self.emotional.extract_all_emotional_features(audio)
        articulatory = self.articulatory.extract_all_articulatory_features(audio)
        style = self.speaking_style.extract_all_style_features(audio)
        
        # Combine all
        all_features.update(prosody)
        all_features.update(breathing)
        all_features.update(spectral)
        all_features.update(temporal)
        all_features.update(vocal)
        all_features.update(emotional)  # NEW
        all_features.update(articulatory)  # NEW
        all_features.update(style)  # NEW
        
        return all_features
    
    def extract_all_features(self, audio):
        """
        Extract both mel spectrogram and human authentication features
        Returns: (mel_spec, features_dict)
        """
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Get all human authentication features
        features = self.extract_all_human_auth_features(audio)
        
        # Ensure all values are floats
        features = {k: float(v) for k, v in features.items()}
        
        return mel_spec, features


# Test
if __name__ == "__main__":
    try:
        from ml_models.audio_processor import AudioProcessor
    except ImportError:
        from audio_processor import AudioProcessor
    
    from pathlib import Path
    
    processor = AudioProcessor()
    extractor = FeatureExtractor()
    
    dataset_path = Path("../dataset/train/human/english")
    audio_files = list(dataset_path.glob("*.mp3")) + list(dataset_path.glob("*.wav"))
    
    if not audio_files:
        for lang in ['tamil', 'hindi', 'malayalam', 'telugu']:
            dataset_path = Path(f"../dataset/train/human/{lang}")
            audio_files = list(dataset_path.glob("*.mp3")) + list(dataset_path.glob("*.wav"))
            if audio_files:
                print(f"✅ Found files in {lang}")
                break
    
    if not audio_files:
        print("❌ No audio files found!")
        exit(1)
    
    test_file = audio_files[0]
    print(f"🎵 Testing with: {test_file}")
    
    try:
        audio = processor.process_from_file(str(test_file))
        print(f"✅ Audio processed: shape={audio.shape}")
        
        mel_spec, features = extractor.extract_all_features(audio)
        
        print(f"✅ Mel Spectrogram shape: {mel_spec.shape}")
        print(f"✅ Number of features: {len(features)}")
        print(f"\n🔍 Human Authentication Features:")
        
        # Show key features
        key_features = [
            'micro_jitter', 'pitch_complexity', 'breath_count',
            'spectral_roughness', 'hnr', 'timing_irregularity'
        ]
        
        for feat in key_features:
            if feat in features:
                print(f"  - {feat}: {features[feat]:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()