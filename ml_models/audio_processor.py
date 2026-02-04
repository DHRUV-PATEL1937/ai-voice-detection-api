"""
Audio Processing Pipeline (Render Compatible)
Uses soundfile instead of pydub/ffmpeg for robust deployment.
"""
import librosa
import numpy as np
import io
import base64
import soundfile as sf

class AudioProcessor:
    """Process audio files for ML pipeline"""
    
    def __init__(self, target_sr=16000, target_duration=5.0):
        """
        Args:
            target_sr: Target sample rate (16kHz is standard for speech)
            target_duration: Target duration in seconds
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
    
    def load_audio_file(self, file_path):
        """Load audio from file path"""
        try:
            # Librosa uses soundfile internally by default, which is safe
            audio, sr = librosa.load(file_path, sr=None) # Load at native SR first
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")
    
    def decode_base64_audio(self, base64_string):
        """
        Decode base64 audio to audio array using soundfile (No ffmpeg required)
        """
        try:
            # 1. Handle Data URI prefix if present (e.g. "data:audio/mp3;base64,...")
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
                
            # 2. Decode base64 to bytes
            audio_bytes = base64.b64decode(base64_string)
            
            # 3. Read into memory buffer
            buffer = io.BytesIO(audio_bytes)
            
            # 4. Load using soundfile
            # ✅ OPTIMIZATION: Read only first ~30 seconds (1.5M frames at 48k)
            # This prevents 5-minute songs from crashing the server
            MAX_FRAMES = 1500000 
            data, sr = sf.read(buffer, stop=MAX_FRAMES)
            
            # 5. Ensure float32 (soundfile might return float64 or int16)
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                
            # 6. Convert Stereo to Mono
            # If shape is (N, 2), average the channels
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            return data, sr
            
        except Exception as e:
            print(f"❌ Audio Decode Error: {e}")
            # Raise error so API catches it properly
            raise ValueError(f"Failed to decode audio: {str(e)}")
    
    def preprocess(self, audio, sr):
        """
        Preprocess audio: resample, normalize, pad/trim
        """
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.target_sr
            )
        
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Trim silence from beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad or trim to fixed target length
        if len(audio) > self.target_length:
            # Trim from center
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        else:
            # Pad with zeros
            pad_length = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')
        
        return audio
    
    def process_from_file(self, file_path):
        """Complete processing pipeline from file"""
        audio, sr = self.load_audio_file(file_path)
        return self.preprocess(audio, sr)
    
    def process_from_base64(self, base64_string):
        """Complete processing pipeline from base64"""
        audio, sr = self.decode_base64_audio(base64_string)
        return self.preprocess(audio, sr)

# Test the processor
if __name__ == "__main__":
    print("🧪 Testing AudioProcessor...")
    processor = AudioProcessor()
    print("✅ AudioProcessor initialized successfully")