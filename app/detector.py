"""
Voice Detection Logic
Loads model and performs inference
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.model import HybridVoiceDetector
from ml_models.audio_processor import AudioProcessor
from ml_models.feature_extractor import FeatureExtractor
from config.settings import MODEL_PATH, NUM_ACOUSTIC_FEATURES

class VoiceDetector:
    """
    Voice Detection Engine
    Loads trained model and performs inference
    """
    
    def __init__(self, model_path=None):
        """
        Initialize detector with trained model
        
        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Initializing detector on {self.device}")
        
        if model_path is None:
            model_path = MODEL_PATH
        
        # Initialize processors
        self.processor = AudioProcessor()
        self.extractor = FeatureExtractor()
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        
        # Create model
        model = HybridVoiceDetector(
            num_acoustic_features=NUM_ACOUSTIC_FEATURES,
            dropout=0.3
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        
        return model
    
    def detect(self, base64_audio: str, language: str) -> dict:
        """
        Detect if audio is AI-generated or human
        
        Args:
            base64_audio: Base64 encoded MP3 audio
            language: Language of the audio
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Process audio
            audio = self.processor.process_from_base64(base64_audio)
            
            # Extract features
            mel_spec, acoustic_features = self.extractor.extract_all_features(audio)
            
            # Prepare tensors
            mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
            
            # Ensure consistent feature count
            sorted_keys = sorted(acoustic_features.keys())
            acoustic_values = [acoustic_features[key] for key in sorted_keys]
            
            # Pad or trim to NUM_ACOUSTIC_FEATURES
            if len(acoustic_values) < NUM_ACOUSTIC_FEATURES:
                acoustic_values += [0.0] * (NUM_ACOUSTIC_FEATURES - len(acoustic_values))
            elif len(acoustic_values) > NUM_ACOUSTIC_FEATURES:
                acoustic_values = acoustic_values[:NUM_ACOUSTIC_FEATURES]
            
            acoustic_tensor = torch.FloatTensor(acoustic_values).unsqueeze(0)
            
            # Move to device
            mel_spec_tensor = mel_spec_tensor.to(self.device)
            acoustic_tensor = acoustic_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(mel_spec_tensor, acoustic_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            # Get results
            classification = "AI_GENERATED" if prediction.item() == 0 else "HUMAN"
            confidence_score = confidence.item()
            
            # Generate explanation
            explanation = self._generate_explanation(
                classification,
                confidence_score,
                acoustic_features
            )
            
            return {
                "status": "success",
                "language": language,
                "classification": classification,
                "confidenceScore": round(confidence_score, 2),
                "explanation": explanation
            }
            
        except Exception as e:
            raise RuntimeError(f"Detection failed: {str(e)}")
    
    def _generate_explanation(self, classification: str, confidence: float, features: dict) -> str:
        """
        Generate human-readable explanation for the classification
        
        Args:
            classification: AI_GENERATED or HUMAN
            confidence: Confidence score
            features: Acoustic features dictionary
            
        Returns:
            Explanation string
        """
        if classification == "AI_GENERATED":
            reasons = []
            
            # Check pitch consistency
            if features.get('pitch_std', 100) < 15:
                reasons.append("unnaturally consistent pitch")
            
            # Check energy stability
            if features.get('energy_std', 0.1) < 0.02:
                reasons.append("overly stable energy levels")
            
            # Check pauses
            if features.get('num_pauses', 10) < 2:
                reasons.append("absence of natural speech pauses")
            
            # Check spectral features
            if features.get('spectral_centroid_std', 1000) < 200:
                reasons.append("monotonous spectral patterns")
            
            if reasons:
                return f"Detected {', '.join(reasons)} characteristic of synthetic speech"
            else:
                return "Audio exhibits spectral and prosodic patterns consistent with AI voice synthesis"
        
        else:  # HUMAN
            reasons = []
            
            # Check for human characteristics
            if features.get('pitch_std', 0) > 25:
                reasons.append("natural pitch variation")
            
            if features.get('num_pauses', 0) > 3:
                reasons.append("organic speech pauses")
            
            if features.get('energy_sudden_changes', 0) > 8:
                reasons.append("natural energy fluctuations")
            
            if features.get('spectral_bandwidth_std', 0) > 300:
                reasons.append("variable spectral characteristics")
            
            if reasons:
                return f"Detected {', '.join(reasons)} indicating authentic human speech"
            else:
                return "Audio characteristics align with natural human voice production patterns"


# Test the detector
if __name__ == "__main__":
    print("🧪 Testing Voice Detector...")
    
    try:
        detector = VoiceDetector()
        print("✅ Detector initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()