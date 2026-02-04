"""
Audio Processing Pipeline (Render Compatible & Fast)
Uses Scipy for resampling to avoid 'samplerate' dependency errors.
"""
import librosa
import numpy as np
import io
import base64
import soundfile as sf
import scipy.signal

class AudioProcessor:
    """Process audio files for ML pipeline"""
    
    def __init__(self, target_sr=16000, target_duration=4.0):
        """
        Args:
            target_sr: Target sample rate (16kHz)
            target_duration: 4.0 seconds (Ultra-fast processing)
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
    
    def load_audio_file(self, file_path):
        """Load audio from file path"""
        try:
            # Librosa uses soundfile internally by default
            # Load at native SR first to avoid librosa's internal resampling
            audio, sr = librosa.load(file_path, sr=None) 
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")

    def decode_base64_audio(self, base64_string):
        try:
            # 1. Handle Data URI
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                
            # 2. Decode
            audio_bytes = base64.b64decode(base64_string)
            buffer = io.BytesIO(audio_bytes)
            
            # 3. Load using soundfile
            # ✅ Read only first 4 seconds (approx 64k samples at 16k, or 192k at 48k)
            # 200,000 frames is a safe upper limit for <5s of audio
            MAX_FRAMES = 200000 
            data, sr = sf.read(buffer, stop=MAX_FRAMES)
            
            # 4. Ensure float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                
            # 5. Mono conversion
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            return data, sr
            
        except Exception as e:
            print(f"❌ Audio Decode Error: {e}")
            raise ValueError(f"Failed to decode audio: {str(e)}")
    
    def preprocess(self, audio, sr):
        """
        Preprocess with Scipy resampling (Bypasses librosa dependency issues)
        """
        # 1. Resample using Scipy (Fast & Robust)
        if sr != self.target_sr:
            # Calculate number of samples after resampling
            number_of_samples = int(len(audio) * float(self.target_sr) / sr)
            # Scipy's resample is faster and doesn't need external C libraries
            audio = scipy.signal.resample(audio, number_of_samples)
        
        # 2. Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # 3. Trim silence
        # We allow librosa here as trim usually works fine without 'samplerate'
        try:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        except:
            pass # Skip trim if it fails, not critical
        
        # 4. Pad or trim to EXACT target length
        if len(audio) > self.target_length:
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        else:
            pad_length = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        return audio
    
    def process_from_file(self, file_path):
        """Complete processing pipeline from file"""
        audio, sr = self.load_audio_file(file_path)
        return self.preprocess(audio, sr)
    
    def process_from_base64(self, base64_string):
        audio, sr = self.decode_base64_audio(base64_string)
        return self.preprocess(audio, sr)

# Test the processor
if __name__ == "__main__":
    print("🧪 Testing AudioProcessor...")
    processor = AudioProcessor()
    print("✅ AudioProcessor initialized successfully")