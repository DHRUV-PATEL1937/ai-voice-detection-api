"""
Audio Processing Pipeline (Ultra-Fast for Render Free Tier)
"""
import librosa
import numpy as np
import io
import base64
import soundfile as sf

class AudioProcessor:
    """Process audio files for ML pipeline"""
    
    def __init__(self, target_sr=16000, target_duration=4.0):
        """
        Args:
            target_sr: Target sample rate (16kHz)
            target_duration: REDUCED to 4.0 seconds for speed
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
    
    def decode_base64_audio(self, base64_string):
        try:
            # 1. Handle Data URI
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                
            # 2. Decode
            audio_bytes = base64.b64decode(base64_string)
            buffer = io.BytesIO(audio_bytes)
            
            # 3. Load using soundfile
            # ✅ ULTRA-OPTIMIZATION: Read only first 4 seconds
            # 16000 Hz * 4 sec = 64,000 samples (plus safety margin)
            # This makes processing instant.
            MAX_FRAMES = 80000 
            data, sr = sf.read(buffer, stop=MAX_FRAMES)
            
            # 4. Ensure float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                
            # 5. Mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            return data, sr
            
        except Exception as e:
            print(f"❌ Audio Decode Error: {e}")
            raise ValueError(f"Failed to decode audio: {str(e)}")
    
    def preprocess(self, audio, sr):
        """
        Preprocess with fast resampling
        """
        # Resample if needed
        if sr != self.target_sr:
            # ✅ OPTIMIZATION: Use 'linear' resampling (Much faster than kaiser_best)
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.target_sr,
                res_type='linear' 
            )
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Trim silence (fast version)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad or trim to EXACT target length
        if len(audio) > self.target_length:
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        else:
            pad_length = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        return audio
    
    def process_from_base64(self, base64_string):
        audio, sr = self.decode_base64_audio(base64_string)
        return self.preprocess(audio, sr)