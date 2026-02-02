"""
Emotional Micro-Expression Features
Humans have natural emotional variations even in neutral speech
"""
import librosa
import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew

class EmotionalFeatureExtractor:
    """Extract emotional micro-expressions from speech"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_emotional_arousal(self, audio):
        """
        Emotional arousal indicators
        Humans show micro-variations in energy that correlate with emotional state
        """
        features = {}
        
        # 1. Energy contour dynamics
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Compute derivatives (how energy changes)
        energy_velocity = np.diff(rms)
        energy_acceleration = np.diff(energy_velocity)
        
        # Statistical moments
        features['energy_velocity_mean'] = np.mean(energy_velocity)
        features['energy_velocity_std'] = np.std(energy_velocity)
        features['energy_acceleration_mean'] = np.mean(energy_acceleration)
        features['energy_acceleration_std'] = np.std(energy_acceleration)
        
        # Kurtosis and skewness (capture distribution shape)
        features['energy_kurtosis'] = float(kurtosis(rms))
        features['energy_skewness'] = float(skew(rms))
        
        # 2. Sudden energy bursts (emphasis, stress)
        # Humans emphasize certain words/syllables
        threshold = np.mean(rms) + 1.5 * np.std(rms)
        bursts = rms > threshold
        
        if np.sum(bursts) > 0:
            burst_durations = []
            in_burst = False
            burst_start = 0
            
            for i, is_burst in enumerate(bursts):
                if is_burst and not in_burst:
                    burst_start = i
                    in_burst = True
                elif not is_burst and in_burst:
                    burst_durations.append(i - burst_start)
                    in_burst = False
            
            features['stress_burst_count'] = len(burst_durations)
            features['stress_burst_avg_duration'] = np.mean(burst_durations) if burst_durations else 0
        else:
            features['stress_burst_count'] = 0
            features['stress_burst_avg_duration'] = 0
        
        return features
    
    def extract_voice_quality_variations(self, audio):
        """
        Voice quality changes during speech
        Humans have natural variations in voice quality (breathy, tense, creaky)
        """
        features = {}
        
        # 1. Spectral tilt (brightness variation over time)
        # Compute STFT
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)
        
        # Spectral tilt for each frame
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        tilts = []
        for frame in magnitude.T:
            # Linear regression of log magnitude vs frequency
            log_mag = np.log(frame + 1e-8)
            
            # Use only meaningful frequencies (100-4000 Hz)
            mask = (freq_bins >= 100) & (freq_bins <= 4000)
            if np.sum(mask) > 10:
                tilt = np.polyfit(freq_bins[mask], log_mag[mask], 1)[0]
                tilts.append(tilt)
        
        if tilts:
            features['spectral_tilt_mean'] = np.mean(tilts)
            features['spectral_tilt_std'] = np.std(tilts)
            features['spectral_tilt_range'] = np.max(tilts) - np.min(tilts)
        else:
            features['spectral_tilt_mean'] = 0
            features['spectral_tilt_std'] = 0
            features['spectral_tilt_range'] = 0
        
        # 2. Cepstral Peak Prominence (CPP) - voice quality measure
        # Higher CPP = more periodic (modal voice)
        # Lower CPP = breathier/creakier
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # CPP approximation using first MFCC coefficient
        cpp_proxy = mfcc[0, :]
        features['voice_quality_mean'] = np.mean(cpp_proxy)
        features['voice_quality_std'] = np.std(cpp_proxy)
        features['voice_quality_variation'] = np.std(cpp_proxy) / (np.abs(np.mean(cpp_proxy)) + 1e-8)
        
        return features
    
    def extract_prosodic_patterns(self, audio):
        """
        Natural prosodic patterns (rhythm, intonation)
        Humans have culturally-influenced prosody
        """
        features = {}
        
        # 1. Pitch contour patterns
        f0 = librosa.yin(audio, fmin=65, fmax=400, sr=self.sr)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 10:
            # Declination (tendency for pitch to fall toward end of utterance)
            # Fit linear trend
            x = np.arange(len(f0_clean))
            trend = np.polyfit(x, f0_clean, 1)[0]
            features['pitch_declination'] = trend
            
            # Pitch reset (humans reset pitch at phrase boundaries)
            # Find local minima and maxima
            from scipy.signal import argrelextrema
            
            local_max = argrelextrema(f0_clean, np.greater, order=5)[0]
            local_min = argrelextrema(f0_clean, np.less, order=5)[0]
            
            features['pitch_peaks_count'] = len(local_max)
            features['pitch_valleys_count'] = len(local_min)
            
            # Peak-to-valley ratio (humans have balanced)
            peak_valley_ratio = len(local_max) / (len(local_min) + 1)
            features['pitch_peak_valley_ratio'] = peak_valley_ratio
            
        else:
            features['pitch_declination'] = 0
            features['pitch_peaks_count'] = 0
            features['pitch_valleys_count'] = 0
            features['pitch_peak_valley_ratio'] = 0
        
        # 2. Rhythm patterns (using onset strength)
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        # Autocorrelation to find rhythmic periodicity
        onset_autocorr = np.correlate(onset_env, onset_env, mode='full')
        onset_autocorr = onset_autocorr[len(onset_autocorr)//2:]
        
        # Find peaks in autocorrelation (indicates rhythm)
        peaks, _ = signal.find_peaks(onset_autocorr, height=np.max(onset_autocorr)*0.3)
        
        features['rhythm_periodicity'] = len(peaks)
        
        if len(peaks) > 1:
            # Inter-peak intervals (rhythm consistency)
            peak_intervals = np.diff(peaks)
            features['rhythm_consistency'] = np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-8)
        else:
            features['rhythm_consistency'] = 0
        
        return features
    
    def extract_all_emotional_features(self, audio):
        """Extract all emotional micro-expression features"""
        all_features = {}
        
        arousal = self.extract_emotional_arousal(audio)
        voice_quality = self.extract_voice_quality_variations(audio)
        prosody = self.extract_prosodic_patterns(audio)
        
        all_features.update(arousal)
        all_features.update(voice_quality)
        all_features.update(prosody)
        
        # Ensure all values are float
        for key in all_features:
            if np.isnan(all_features[key]) or np.isinf(all_features[key]):
                all_features[key] = 0.0
            all_features[key] = float(all_features[key])
        
        return all_features