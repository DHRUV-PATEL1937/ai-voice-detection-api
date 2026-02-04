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
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Initializing human authenticator on {self.device}")
        
        # If no path provided, search for model
        if model_path is None:
            possible_paths = [
                # First check the standard location
                Path('saved_models/best_model.pth'),
                
                # Then check human_authenticator folders (most recent first)
                *sorted(
                    Path('saved_models').glob('human_authenticator_*/best_model.pth'),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                ),
                
                # Deployment locations
                Path('best_model.pth'),
                Path('/opt/render/project/src/saved_models/best_model.pth'),
                Path('/opt/render/project/src/best_model.pth'),
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    print(f"✅ Found model at: {model_path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "Model file not found! Please ensure best_model.pth exists in saved_models/"
                )
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at: {model_path}")
        
        print(f"📂 Loading model from: {model_path}")
        
        self.processor = AudioProcessor()
        self.extractor = FeatureExtractor()
        
        # Load model checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"✅ Checkpoint loaded")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Get number of features from checkpoint
        try:
            num_features = checkpoint['config']['num_acoustic_features']
            print(f"✅ Model expects {num_features} acoustic features")
        except KeyError:
            print(f"⚠️  'num_acoustic_features' not in checkpoint config")
            raise
        
        # Create model
        try:
            self.model = HumanVoiceAuthenticator(
                num_acoustic_features=num_features,
                dropout=0.3
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # ✅ OPTIMIZATION: Apply Dynamic Quantization for 2x CPU Speedup
            if self.device.type == 'cpu':
                print("⚡ Applying dynamic quantization for CPU speedup...")
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, 
                        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, 
                        dtype=torch.qint8
                    )
                    print("⚡ Model quantized successfully")
                except Exception as q_err:
                    print(f"⚠️ Quantization skipped: {q_err}")

            print(f"✅ Model initialized")
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            raise RuntimeError(f"Failed to initialize model: {e}")
        
        # Load human centroid and threshold
        try:
            self.human_centroid = checkpoint['human_centroid'].to(self.device)
            self.threshold = checkpoint.get('threshold', 0.5)
            print(f"✅ Human centroid loaded")
            print(f"✅ Detection threshold: {self.threshold:.4f}")
        except Exception as e:
            print(f"❌ Failed to load centroid: {e}")
            raise
        
        print(f"✅ Detector ready!")
    
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
                
                # Decision
                is_human = distance < self.threshold
                classification = "HUMAN" if is_human else "AI_GENERATED"
                
                # IMPROVED CONFIDENCE CALCULATION
                if is_human:
                    normalized_distance = min(distance / self.threshold, 1.0)
                    confidence = 100 - (normalized_distance * 35)  # 100% to 65%
                else:
                    excess_distance = distance - self.threshold
                    normalized_excess = min(excess_distance / self.threshold, 1.0)
                    confidence = 65 + (normalized_excess * 35)  # 65% to 100%
                
                # Ensure confidence is in valid range
                confidence = max(60, min(confidence, 100))
                
                print(f"📊 Distance: {distance:.4f}, Threshold: {self.threshold:.4f}, Confidence: {confidence:.1f}%")
            
            # Generate explanation
            explanation = self._generate_explanation(
                classification,
                distance,
                self.threshold,
                acoustic_features,
                confidence
            )
            
            return {
                "status": "success",
                "language": language,
                "classification": classification,
                "confidenceScore": round(float(confidence) / 100, 2),  # Return as 0.0-1.0
                "explanation": explanation
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Detection failed: {str(e)}")
    
    def _generate_explanation(self, classification, distance, threshold, features, confidence):
        """Generate human-friendly explanation"""
        
        if classification == "HUMAN":
            # Identify strong human characteristics
            human_traits = []
            
            if features.get('micro_jitter', 0) > 0.5:
                human_traits.append("natural pitch variations")
            
            if features.get('breath_count', 0) > 2:
                human_traits.append("authentic breathing patterns")
            elif features.get('breath_count', 0) > 0:
                human_traits.append("breathing sounds")
            
            if features.get('spectral_roughness', 0) > 100:
                human_traits.append("natural vocal texture")
            
            if features.get('timing_irregularity', 0) > 0.1:
                human_traits.append("human timing patterns")
            
            if features.get('formant_variability', 0) > 0.05:
                human_traits.append("natural voice modulation")
            
            # Generate explanation based on traits found
            if len(human_traits) >= 3:
                traits_text = ", ".join(human_traits[:3])
                return f"Voice authenticated with {traits_text} characteristic of human speech"
            elif len(human_traits) >= 1:
                traits_text = " and ".join(human_traits)
                return f"Voice shows {traits_text} typical of human speakers"
            else:
                return "Voice matches the acoustic profile of authentic human speech"
        
        else:
            # AI Generated - Focus on what's missing or unnatural
            ai_indicators = []
            
            if features.get('micro_jitter', 0) < 0.3:
                ai_indicators.append("unnaturally stable pitch patterns")
            
            if features.get('breath_count', 0) == 0:
                ai_indicators.append("absence of breathing sounds")
            
            if features.get('spectral_roughness', 0) < 50:
                ai_indicators.append("overly smooth vocal quality")
            
            if features.get('hnr', float('inf')) > 20:
                ai_indicators.append("artificially perfect harmonics")
            
            if features.get('timing_irregularity', 0) < 0.05:
                ai_indicators.append("mechanically precise timing")
            
            if features.get('shimmer', 0) < 0.02:
                ai_indicators.append("suspiciously consistent amplitude")
            
            # Generate explanation based on AI indicators
            if len(ai_indicators) >= 3:
                traits_text = ", ".join(ai_indicators[:3])
                return f"Voice exhibits {traits_text}, indicating AI synthesis"
            elif len(ai_indicators) >= 1:
                traits_text = " and ".join(ai_indicators[:2])
                return f"Detection based on {traits_text} not typical of human speech"
            else:
                return "Voice lacks the natural imperfections and variability characteristic of human speakers"