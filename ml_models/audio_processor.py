"""
Audio Processing Pipeline
Handles: Base64 decoding, preprocessing, normalization
"""
import librosa
import numpy as np
from pydub import AudioSegment
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
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")
    
    def decode_base64_audio(self, base64_string):
        """
        Decode base64 MP3 to audio array
        
        Args:
            base64_string: Base64 encoded audio
            
        Returns:
            audio: numpy array
            sr: sample rate
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_string)
            
            # Load with pydub (handles MP3)
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format="mp3"
            )
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Handle stereo
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)  # Convert to mono
            
            # Normalize
            samples = samples / (2**15)  # 16-bit audio
            
            sr = audio.frame_rate
            
            return samples, sr
            
        except Exception as e:
            raise ValueError(f"Error decoding base64 audio: {e}")
    
    def preprocess(self, audio, sr):
        """
        Preprocess audio: resample, normalize, pad/trim
        
        Args:
            audio: numpy array
            sr: current sample rate
            
        Returns:
            processed audio
        """
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(
                y=audio.astype(float),
                orig_sr=sr,
                target_sr=self.target_sr
            )
        
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Trim silence from beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad or trim to target length
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
    processor = AudioProcessor()
    
    # Test with a file
    test_file = "dataset/train/human/english/sample.mp3"  # Use your actual file
    try:
        audio = processor.process_from_file(test_file)
        print(f"✅ Processed audio shape: {audio.shape}")
        print(f"✅ Audio duration: {len(audio) / processor.target_sr:.2f} seconds")
    except Exception as e:
        print(f"❌ Error: {e}")