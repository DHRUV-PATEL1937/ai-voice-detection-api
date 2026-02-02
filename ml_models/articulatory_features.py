"""
Articulatory Features - Physical speech production characteristics
"""
import librosa
import numpy as np
from scipy import signal

class ArticulatoryFeatureExtractor:
    """Extract features related to physical articulation"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_consonant_precision(self, audio):
        """
        Consonant production characteristics
        Humans have physical limitations in articulation
        """
        features = {}
        
        # 1. High-frequency transients (consonants like /t/, /k/, /p/)
        # Use zero-crossing rate as proxy
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
        
        # Find high ZCR regions (likely consonants)
        high_zcr_threshold = np.mean(zcr) + np.std(zcr)
        consonant_regions = zcr > high_zcr_threshold
        
        # Count consonant-like events
        features['consonant_count'] = np.sum(np.diff(consonant_regions.astype(int)) == 1)
        
        # 2. Stop consonant detection (sharp onset)
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        # Sharp onsets indicate stop consonants
        sharp_onsets = onset_env > (np.mean(onset_env) + 2 * np.std(onset_env))
        features['stop_consonant_count'] = np.sum(sharp_onsets)
        
        # 3. Fricative detection (sustained high-frequency noise)
        # Spectral flatness in high frequencies
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        
        # High frequencies (2000 Hz and above)
        freqs = librosa.fft_frequencies(sr=self.sr)
        high_freq_mask = freqs >= 2000
        
        flatness_values = []
        for frame in magnitude.T:
            high_freq_frame = frame[high_freq_mask]
            if len(high_freq_frame) > 0:
                # Spectral flatness (geometric mean / arithmetic mean)
                geo_mean = np.exp(np.mean(np.log(high_freq_frame + 1e-8)))
                arith_mean = np.mean(high_freq_frame)
                flatness = geo_mean / (arith_mean + 1e-8)
                flatness_values.append(flatness)
        
        if flatness_values:
            features['fricative_indicator'] = np.mean(flatness_values)
        else:
            features['fricative_indicator'] = 0
        
        return features
    
    def extract_formant_dynamics(self, audio):
        """
        Formant transitions - how vowels connect
        Humans have smooth but imperfect transitions
        """
        features = {}
        
        # Extract MFCCs (proxies for formants)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # First 4 MFCCs relate to first 2 formants (F1, F2)
        f1_proxy = mfcc[1, :]
        f2_proxy = mfcc[2, :]
        
        # 1. Formant transition rate
        f1_velocity = np.diff(f1_proxy)
        f2_velocity = np.diff(f2_proxy)
        
        features['f1_transition_rate'] = np.mean(np.abs(f1_velocity))
        features['f2_transition_rate'] = np.mean(np.abs(f2_velocity))
        
        # 2. Formant overshooting (humans overshoot targets)
        # Measure oscillation in formant tracks
        f1_peaks, _ = signal.find_peaks(f1_proxy)
        f1_valleys, _ = signal.find_peaks(-f1_proxy)
        
        features['f1_oscillations'] = len(f1_peaks) + len(f1_valleys)
        
        # 3. Coarticulation (sounds blend into each other)
        # Measured by smoothness of transitions
        f1_smoothness = np.std(np.diff(f1_velocity))
        f2_smoothness = np.std(np.diff(f2_velocity))
        
        features['coarticulation_f1'] = f1_smoothness
        features['coarticulation_f2'] = f2_smoothness
        
        return features
    
    def extract_vocal_effort_markers(self, audio):
        """
        Markers of vocal effort and fatigue
        Humans show subtle signs of physical effort
        """
        features = {}
        
        # 1. Harmonic richness variation
        # Extract harmonics and noise
        harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)
        
        # Frame-level harmonic-to-noise ratio
        frame_length = 2048
        hop_length = 512
        
        hnr_values = []
        
        for i in range(0, len(audio) - frame_length, hop_length):
            h_frame = harmonic[i:i+frame_length]
            p_frame = percussive[i:i+frame_length]
            
            h_energy = np.sum(h_frame ** 2)
            p_energy = np.sum(p_frame ** 2)
            
            if p_energy > 1e-8:
                hnr = 10 * np.log10(h_energy / p_energy)
                hnr_values.append(hnr)
        
        if hnr_values:
            features['hnr_mean'] = np.mean(hnr_values)
            features['hnr_std'] = np.std(hnr_values)
            features['hnr_variation'] = np.std(hnr_values) / (np.abs(np.mean(hnr_values)) + 1e-8)
        else:
            features['hnr_mean'] = 0
            features['hnr_std'] = 0
            features['hnr_variation'] = 0
        
        # 2. Vocal tension indicators
        # High-frequency emphasis (tense voice)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        
        features['vocal_tension_mean'] = np.mean(spectral_centroid)
        features['vocal_tension_std'] = np.std(spectral_centroid)
        
        # 3. Breathiness tracking
        # Ratio of high-frequency noise to low-frequency energy
        D = librosa.stft(audio)
        magnitude = np.abs(D)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        low_freq_mask = (freqs >= 100) & (freqs <= 1000)
        high_freq_mask = (freqs >= 2000) & (freqs <= 4000)
        
        breathiness_values = []
        for frame in magnitude.T:
            low_energy = np.sum(frame[low_freq_mask] ** 2)
            high_energy = np.sum(frame[high_freq_mask] ** 2)
            
            if low_energy > 1e-8:
                breathiness = high_energy / low_energy
                breathiness_values.append(breathiness)
        
        if breathiness_values:
            features['breathiness_mean'] = np.mean(breathiness_values)
            features['breathiness_variation'] = np.std(breathiness_values)
        else:
            features['breathiness_mean'] = 0
            features['breathiness_variation'] = 0
        
        return features
    
    def extract_all_articulatory_features(self, audio):
        """Extract all articulatory features"""
        all_features = {}
        
        consonant = self.extract_consonant_precision(audio)
        formant = self.extract_formant_dynamics(audio)
        effort = self.extract_vocal_effort_markers(audio)
        
        all_features.update(consonant)
        all_features.update(formant)
        all_features.update(effort)
        
        # Ensure all values are float
        for key in all_features:
            if np.isnan(all_features[key]) or np.isinf(all_features[key]):
                all_features[key] = 0.0
            all_features[key] = float(all_features[key])
        
        return all_features