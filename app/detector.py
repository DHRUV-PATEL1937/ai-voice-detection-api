"""
Human Voice Authenticator - Detection Logic
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.human_voice_authenticator import HumanVoiceAuthenticator
from ml_models.audio_processor import AudioProcessor
from ml_models.feature_extractor import FeatureExtractor

class VoiceDetector:
    """Authenticates human voices, detects anything else as AI"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Initializing human authenticator on {self.device}")
        
        # Try to find model in multiple locations
        possible_model_paths = [
            Path('saved_models/best_model.pth'),
            Path('best_model.pth'),
            Path('/opt/render/project/src/saved_models/best_model.pth'),
            Path('/opt/render/project/src/best_model.pth'),
        ]
        
        model_path = None
        for path in possible_model_paths:
            if path.exists():
                model_path = path
                print(f"✅ Found model at: {model_path}")
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"Model file not found! Searched:\n" + 
                "\n".join([f"  - {p}" for p in possible_model_paths])
            )
        
        self.processor = AudioProcessor()
        self.extractor = FeatureExtractor()
        
        # Load model checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"✅ Checkpoint loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Get number of features from checkpoint
        num_features = checkpoint['config']['num_acoustic_features']
        print(f"✅ Model expects {num_features} acoustic features")
        
        # Create model
        try:
            self.model = HumanVoiceAuthenticator(
                num_acoustic_features=num_features,
                dropout=0.3
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
        
        # Load human centroid and threshold
        self.human_centroid = checkpoint['human_centroid'].to(self.device)
        self.threshold = checkpoint.get('threshold', 0.5)
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Detection threshold: {self.threshold:.4f}")
    
    def detect(self, base64_audio: str, language: str) -> dict:
        """Authenticate if voice is human"""
        
        try:
            # Process audio
            audio = self.processor.process_from_base64(base64_audio)
            
            # Extract features
            mel_spec, acoustic_features = self.extractor.extract_all_features(audio)
            
            # Prepare tensors
            mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
            
            sorted_keys = sorted(acoustic_features.keys())
            acoustic_values = [acoustic_features[key] for key in sorted_keys]
            
            num_features = self.model.acoustic_mlp[0].in_features
            
            if len(acoustic_values) < num_features:
                acoustic_values += [0.0] * (num_features - len(acoustic_values))
            elif len(acoustic_values) > num_features:
                acoustic_values = acoustic_values[:num_features]
            
            acoustic_tensor = torch.FloatTensor(acoustic_values).unsqueeze(0)
            
            # Move to device
            mel_spec_tensor = mel_spec_tensor.to(self.device)
            acoustic_tensor = acoustic_tensor.to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model(mel_spec_tensor, acoustic_tensor)
                
                # Compute distance from human centroid
                distance = torch.norm(embedding - self.human_centroid, dim=1).item()
                
                # Compute confidence
                authenticity_score = np.exp(-distance)
                
                # Decision
                is_human = distance < self.threshold
                classification = "HUMAN" if is_human else "AI_GENERATED"
                confidence = authenticity_score if is_human else (1 - authenticity_score)
            
            # Generate explanation
            explanation = self._generate_explanation(
                classification,
                distance,
                self.threshold,
                acoustic_features
            )
            
            return {
                "status": "success",
                "language": language,
                "classification": classification,
                "confidenceScore": round(float(confidence), 2),
                "explanation": explanation
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Detection failed: {str(e)}")
    
    def _generate_explanation(self, classification, distance, threshold, features):
        """Generate explanation based on distance from human centroid"""
        
        if classification == "HUMAN":
            reasons = []
            
            if features.get('micro_jitter', 0) > 0.5:
                reasons.append("natural pitch micro-variations")
            
            if features.get('breath_count', 0) > 2:
                reasons.append("authentic breathing patterns")
            
            if features.get('spectral_roughness', 0) > 100:
                reasons.append("natural vocal tract noise")
            
            if features.get('timing_irregularity', 0) > 0.1:
                reasons.append("human timing irregularities")
            
            if reasons:
                return f"Voice authenticated with {', '.join(reasons)} characteristic of human speech"
            else:
                return f"Voice matches human vocal characteristics (distance: {distance:.3f})"
        
        else:
            reasons = []
            
            if features.get('micro_jitter', 0) < 0.3:
                reasons.append("unnaturally stable pitch")
            
            if features.get('breath_count', 0) == 0:
                reasons.append("absence of breathing sounds")
            
            if features.get('spectral_roughness', 0) < 50:
                reasons.append("overly clean spectral characteristics")
            
            if features.get('hnr', float('inf')) > 20:
                reasons.append("too-perfect harmonic structure")
            
            if reasons:
                return f"Failed human authentication due to {', '.join(reasons)}"
            else:
                return f"Voice does not match human vocal patterns (distance: {distance:.3f} > threshold: {threshold:.3f})"