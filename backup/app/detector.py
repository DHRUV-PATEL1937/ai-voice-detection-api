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
from config.settings import MODEL_PATH, NUM_ACOUSTIC_FEATURES

class VoiceDetector:
    """Authenticates human voices, detects anything else as AI"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Initializing human authenticator on {self.device}")
        
        if model_path is None:
            model_path = MODEL_PATH
        
        self.processor = AudioProcessor()
        self.extractor = FeatureExtractor()
        
        # Load model and centroid
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = HumanVoiceAuthenticator(
            num_acoustic_features=NUM_ACOUSTIC_FEATURES,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load human centroid and threshold
        self.human_centroid = checkpoint['human_centroid'].to(self.device)
        self.threshold = checkpoint.get('threshold', 0.5)
        
        print(f"✅ Model loaded from {model_path}")
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
            
            if len(acoustic_values) < NUM_ACOUSTIC_FEATURES:
                acoustic_values += [0.0] * (NUM_ACOUSTIC_FEATURES - len(acoustic_values))
            elif len(acoustic_values) > NUM_ACOUSTIC_FEATURES:
                acoustic_values = acoustic_values[:NUM_ACOUSTIC_FEATURES]
            
            acoustic_tensor = torch.FloatTensor(acoustic_values).unsqueeze(0)
            
            # Move to device
            mel_spec_tensor = mel_spec_tensor.to(self.device)
            acoustic_tensor = acoustic_tensor.to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model(mel_spec_tensor, acoustic_tensor)
                
                # Compute distance from human centroid
                distance = torch.norm(embedding - self.human_centroid, dim=1).item()
                
                # Decision
                is_human = distance < self.threshold
                classification = "HUMAN" if is_human else "AI_GENERATED"
                
                # Sigmoid confidence calculation
                steepness = 5.0 / self.threshold
                sigmoid_score = 1.0 / (1.0 + np.exp(steepness * (distance - self.threshold)))
                
                if is_human:
                    confidence_percentage = 60 + (40 * sigmoid_score)
                else:
                    confidence_percentage = 60 + (40 * (1 - sigmoid_score))
                
                confidence_percentage = max(60, min(100, confidence_percentage))
            
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
                "confidenceScore": round(float(confidence_percentage / 100), 4),
                "explanation": explanation
            }
            
        except Exception as e:
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
            
            if distance < threshold * 0.5:
                confidence_desc = "Strong human voice signature detected"
            elif distance < threshold * 0.75:
                confidence_desc = "Clear human vocal characteristics"
            else:
                confidence_desc = "Human voice authenticated"
            
            if reasons:
                return f"{confidence_desc} with {', '.join(reasons[:3])}"
            else:
                return f"{confidence_desc} (authenticity distance: {distance:.3f})"
        
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
            
            if distance > threshold * 1.5:
                confidence_desc = "Strong AI signature detected"
            elif distance > threshold * 1.2:
                confidence_desc = "Clear AI-generated characteristics"
            else:
                confidence_desc = "Failed human authentication"
            
            if reasons:
                return f"{confidence_desc}: {', '.join(reasons[:3])}"
            else:
                return f"{confidence_desc} (deviation from human pattern: {distance:.3f})"