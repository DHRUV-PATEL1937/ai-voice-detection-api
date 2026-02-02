"""
Speaking Style Features - Natural human speech patterns
"""
import librosa
import numpy as np
from scipy import signal

class SpeakingStyleExtractor:
    """Extract natural speaking style characteristics"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def extract_hesitation_patterns(self, audio):
        """
        Natural hesitations and disfluencies
        Humans pause, repeat, self-correct
        """
        features = {}
        
        # 1. Pause patterns
        rms = librosa.feature.rms(y=audio)[0]
        
        # Detect pauses (low energy regions)
        pause_threshold = np.mean(rms) * 0.1
        is_pause = rms < pause_threshold
        
        # Find pause durations
        pause_durations = []
        in_pause = False
        pause_start = 0
        
        for i, paused in enumerate(is_pause):
            if paused and not in_pause:
                pause_start = i
                in_pause = True
            elif not paused and in_pause:
                duration = i - pause_start
                pause_durations.append(duration)
                in_pause = False
        
        if pause_durations:
            features['pause_count'] = len(pause_durations)
            features['pause_mean_duration'] = np.mean(pause_durations)
            features['pause_std_duration'] = np.std(pause_durations)
            
            # Short pauses (filled pauses like "uh", "um")
            short_pauses = [d for d in pause_durations if d < 10]
            features['short_pause_count'] = len(short_pauses)
            
            # Long pauses (thinking pauses)
            long_pauses = [d for d in pause_durations if d >= 10]
            features['long_pause_count'] = len(long_pauses)
        else:
            features['pause_count'] = 0
            features['pause_mean_duration'] = 0
            features['pause_std_duration'] = 0
            features['short_pause_count'] = 0
            features['long_pause_count'] = 0
        
        # 2. False starts and repetitions
        # Detected by similar energy/spectral patterns close together
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        # Find strong onsets
        strong_onsets = signal.find_peaks(onset_env, height=np.mean(onset_env) + np.std(onset_env))[0]
        
        # Check for closely spaced onsets (potential repetitions)
        if len(strong_onsets) > 1:
            onset_intervals = np.diff(strong_onsets)
            
            # Very close onsets might indicate repetition/false start
            close_onsets = np.sum(onset_intervals < 20)
            features['potential_repetitions'] = close_onsets
        else:
            features['potential_repetitions'] = 0
        
        return features
    
    def extract_speech_rate_variation(self, audio):
        """
        Natural variation in speech rate
        Humans speed up and slow down naturally
        """
        features = {}
        
        # Syllable-like units detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        
        # Smooth onset envelope
        from scipy.ndimage import gaussian_filter1d
        onset_smooth = gaussian_filter1d(onset_env, sigma=2)
        
        # Find peaks (syllable-like events)
        peaks, _ = signal.find_peaks(onset_smooth, distance=5)
        
        if len(peaks) > 3:
            # Compute local speech rates (syllables per unit time)
            window_size = 50  # frames
            
            local_rates = []
            for i in range(0, len(onset_smooth) - window_size, window_size // 2):
                window_peaks = peaks[(peaks >= i) & (peaks < i + window_size)]
                rate = len(window_peaks)
                local_rates.append(rate)
            
            if local_rates:
                features['speech_rate_mean'] = np.mean(local_rates)
                features['speech_rate_std'] = np.std(local_rates)
                features['speech_rate_cv'] = np.std(local_rates) / (np.mean(local_rates) + 1e-8)
                
                # Acceleration (change in rate)
                rate_changes = np.abs(np.diff(local_rates))
                features['speech_rate_acceleration'] = np.mean(rate_changes)
            else:
                features['speech_rate_mean'] = 0
                features['speech_rate_std'] = 0
                features['speech_rate_cv'] = 0
                features['speech_rate_acceleration'] = 0
        else:
            features['speech_rate_mean'] = 0
            features['speech_rate_std'] = 0
            features['speech_rate_cv'] = 0
            features['speech_rate_acceleration'] = 0
        
        return features
    
    def extract_emphasis_patterns(self, audio):
        """
        How humans emphasize words
        Natural stress and prominence patterns
        """
        features = {}
        
        # 1. Dynamic range (difference between loudest and quietest)
        rms = librosa.feature.rms(y=audio)[0]
        rms_db = librosa.amplitude_to_db(rms)
        
        features['dynamic_range'] = np.max(rms_db) - np.min(rms_db)
        
        # 2. Prominence peaks (emphasized words)
        # Combine energy and pitch for prominence
        f0 = librosa.yin(audio, fmin=65, fmax=400, sr=self.sr)
        
        # Normalize both
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
        f0_norm = (f0 - np.nanmin(f0)) / (np.nanmax(f0) - np.nanmin(f0) + 1e-8)
        f0_norm = np.nan_to_num(f0_norm)
        
        # Ensure same length
        min_len = min(len(rms_norm), len(f0_norm))
        rms_norm = rms_norm[:min_len]
        f0_norm = f0_norm[:min_len]
        
        # Combined prominence
        prominence = 0.6 * rms_norm + 0.4 * f0_norm
        
        # Find prominence peaks
        prom_peaks, properties = signal.find_peaks(
            prominence,
            height=0.6,
            distance=20
        )
        
        features['emphasis_count'] = len(prom_peaks)
        
        if len(prom_peaks) > 1:
            # Spacing between emphasized words
            emphasis_intervals = np.diff(prom_peaks)
            features['emphasis_spacing_mean'] = np.mean(emphasis_intervals)
            features['emphasis_spacing_std'] = np.std(emphasis_intervals)
        else:
            features['emphasis_spacing_mean'] = 0
            features['emphasis_spacing_std'] = 0
        
        # 3. Micro-emphasis (subtle stress)
        # Small peaks that don't cross prominence threshold
        small_peaks, _ = signal.find_peaks(
            prominence,
            height=0.4,
            distance=10
        )
        
        features['micro_emphasis_count'] = len(small_peaks) - len(prom_peaks)
        
        return features
    
    def extract_all_style_features(self, audio):
        """Extract all speaking style features"""
        all_features = {}
        
        hesitation = self.extract_hesitation_patterns(audio)
        speech_rate = self.extract_speech_rate_variation(audio)
        emphasis = self.extract_emphasis_patterns(audio)
        
        all_features.update(hesitation)
        all_features.update(speech_rate)
        all_features.update(emphasis)
        
        # Ensure all values are float
        for key in all_features:
            if np.isnan(all_features[key]) or np.isinf(all_features[key]):
                all_features[key] = 0.0
            all_features[key] = float(all_features[key])
        
        return all_features