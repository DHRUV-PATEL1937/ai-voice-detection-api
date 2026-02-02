"""
Human Voice Authentication Features
Detects micro-level human characteristics that AI cannot replicate
"""
import librosa
import numpy as np
from scipy import signal, stats
from scipy.fft import fft

class HumanAuthenticator:
    """Extract features that prove human authenticity"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_micro_prosody(self, audio):
        """
        Micro-prosodic features - tiny variations humans make unconsciously
        AI voices are TOO smooth at millisecond level
        """
        features = {}
        
        # Extract pitch
        f0 = librosa.yin(audio, fmin=65, fmax=400, sr=self.sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 10:
            # 1. Micro-jitter (frame-to-frame pitch variation)
            # Humans have natural jitter, AI is too stable
            frame_diffs = np.abs(np.diff(f0_clean))
            features['micro_jitter'] = np.mean(frame_diffs)
            features['jitter_variance'] = np.var(frame_diffs)
            
            # 2. Shimmer (amplitude variation)
            rms = librosa.feature.rms(y=audio)[0]
            shimmer = np.abs(np.diff(rms)) / (np.mean(rms) + 1e-8)
            features['micro_shimmer'] = np.mean(shimmer)
            
            # 3. Pitch contour complexity (humans have complex curves)
            # Use second derivative to measure curvature
            if len(f0_clean) > 20:
                pitch_velocity = np.diff(f0_clean)
                pitch_acceleration = np.diff(pitch_velocity)
                features['pitch_complexity'] = np.std(pitch_acceleration)
            else:
                features['pitch_complexity'] = 0
        else:
            features['micro_jitter'] = 0
            features['jitter_variance'] = 0
            features['micro_shimmer'] = 0
            features['pitch_complexity'] = 0
        
        return features
    
    def extract_breath_signatures(self, audio):
        """
        Detect natural breathing patterns
        AI voices don't breathe naturally
        """
        features = {}
        
        # Detect low-energy regions (potential breaths/pauses)
        rms = librosa.feature.rms(y=audio)[0]
        threshold = np.mean(rms) * 0.1
        
        # Find breath-like regions (low energy but not complete silence)
        breath_mask = (rms < threshold) & (rms > 0)
        
        # Characteristics of breathing
        if np.sum(breath_mask) > 0:
            # Breath frequency (humans breathe regularly)
            breath_intervals = np.diff(np.where(np.diff(breath_mask.astype(int)) == 1)[0])
            if len(breath_intervals) > 0:
                features['breath_regularity'] = np.std(breath_intervals)
                features['breath_count'] = len(breath_intervals)
            else:
                features['breath_regularity'] = 0
                features['breath_count'] = 0
        else:
            features['breath_regularity'] = 0
            features['breath_count'] = 0
        
        # Natural pause characteristics
        silent_regions = rms < (np.mean(rms) * 0.05)
        features['natural_pause_ratio'] = np.sum(silent_regions) / len(rms)
        
        return features
    
    def extract_spectral_irregularities(self, audio):
        """
        Spectral irregularities - humans have natural noise
        AI synthesis is too clean in frequency domain
        """
        features = {}
        
        # STFT
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        
        # 1. Spectral roughness (high-frequency noise)
        # Humans have natural breathiness/roughness
        high_freq_energy = np.sum(magnitude[magnitude.shape[0]//2:, :], axis=0)
        features['spectral_roughness'] = np.std(high_freq_energy)
        
        # 2. Harmonic deviation
        # Humans have slight harmonic imperfections
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(audio**2) + 1e-8)
        features['harmonic_purity'] = harmonic_ratio  # AI is TOO pure
        
        # 3. Spectral entropy (randomness in spectrum)
        # Humans have more entropy (natural variations)
        spectral_entropy = []
        for frame in magnitude.T:
            # Normalize
            prob = frame / (np.sum(frame) + 1e-8)
            # Calculate entropy
            entropy = -np.sum(prob * np.log2(prob + 1e-8))
            spectral_entropy.append(entropy)
        
        features['spectral_entropy_mean'] = np.mean(spectral_entropy)
        features['spectral_entropy_std'] = np.std(spectral_entropy)
        
        return features
    
    def extract_temporal_irregularities(self, audio):
        """
        Temporal micro-patterns - humans are inconsistent
        AI has too-perfect timing
        """
        features = {}
        
        # Speech rate variability
        # Detect syllable-like events using onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        # Find peaks (syllables/words)
        peaks, _ = signal.find_peaks(onset_env, height=np.mean(onset_env))
        
        if len(peaks) > 2:
            # Inter-syllable intervals
            intervals = np.diff(peaks)
            
            # Humans have variable timing
            features['timing_irregularity'] = np.std(intervals) / (np.mean(intervals) + 1e-8)
            features['timing_cv'] = stats.variation(intervals) if len(intervals) > 0 else 0
        else:
            features['timing_irregularity'] = 0
            features['timing_cv'] = 0
        
        # Energy fluctuation patterns
        rms = librosa.feature.rms(y=audio)[0]
        
        # Detrend to remove overall envelope
        detrended = signal.detrend(rms)
        
        # High-frequency energy fluctuations (nervousness, natural variations)
        fft_energy = np.abs(fft(detrended))
        high_freq_fluctuation = np.sum(fft_energy[len(fft_energy)//4:])
        features['energy_micro_fluctuation'] = high_freq_fluctuation
        
        return features
    
    def extract_vocal_tract_artifacts(self, audio):
        """
        Physical vocal tract artifacts
        Real vocal tracts have unique resonances and imperfections
        """
        features = {}
        
        # Formants (vocal tract resonances)
        # Humans have slight formant wobble
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # First 3 MFCCs relate to formants
        for i in range(3):
            formant_track = mfccs[i, :]
            # Measure micro-variations
            features[f'formant_{i}_micro_var'] = np.std(np.diff(formant_track))
        
        # Vocal fry detection (humans sometimes have vocal fry)
        # Low frequency irregularities
        low_freq_energy = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features['vocal_fry_indicator'] = np.sum(low_freq_energy < 200)
        
        return features
    
    def extract_all_human_features(self, audio):
        """Extract all human authentication features"""
        
        all_features = {}
        
        # Micro-prosody
        prosody = self.extract_micro_prosody(audio)
        all_features.update(prosody)
        
        # Breathing
        breath = self.extract_breath_signatures(audio)
        all_features.update(breath)
        
        # Spectral irregularities
        spectral = self.extract_spectral_irregularities(audio)
        all_features.update(spectral)
        
        # Temporal patterns
        temporal = self.extract_temporal_irregularities(audio)
        all_features.update(temporal)
        
        # Vocal tract
        vocal = self.extract_vocal_tract_artifacts(audio)
        all_features.update(vocal)
        
        return all_features