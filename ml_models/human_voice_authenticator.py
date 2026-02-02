"""
Human Voice Authenticator - One-Class Classification
Only trains on HUMAN voices, detects anything else as AI
"""
import torch
import torch.nn as nn
import numpy as np

class HumanVoiceAuthenticator(nn.Module):
    """
    One-class classifier for human voice authentication
    Trained ONLY on human voices
    """
    def __init__(self, num_acoustic_features, dropout=0.3):
        super(HumanVoiceAuthenticator, self).__init__()
        
        # CNN for mel spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten()
        )
        
        # MLP for acoustic features
        self.acoustic_mlp = nn.Sequential(
            nn.Linear(num_acoustic_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Feature fusion and encoding
        self.encoder = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Embedding space
        )
        
        # Decision boundary (distance from human centroid)
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
    def forward(self, mel_spec, acoustic_features):
        """
        Returns: embedding and distance from human centroid
        """
        # Extract features
        cnn_features = self.cnn(mel_spec)
        acoustic_output = self.acoustic_mlp(acoustic_features)
        
        # Combine
        combined = torch.cat([cnn_features, acoustic_output], dim=1)
        
        # Get embedding
        embedding = self.encoder(combined)
        
        return embedding
    
    def compute_authenticity_score(self, embedding, human_centroid):
        """
        Compute how "human-like" the voice is
        Lower distance = more human
        """
        # Euclidean distance from human centroid
        distance = torch.norm(embedding - human_centroid, dim=1)
        
        # Convert to probability (closer = higher probability of human)
        authenticity = torch.exp(-distance)
        
        return authenticity, distance