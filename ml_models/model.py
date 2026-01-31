"""
Hybrid Neural Network for AI Voice Detection
Combines CNN (for spectrograms) + MLP (for acoustic features)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    """
    CNN for processing mel spectrograms
    Input: (batch, 1, 128, time_steps)
    Output: feature vector
    """
    def __init__(self, dropout=0.3):
        super(SpectrogramCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # x shape: (batch, 1, 128, time_steps)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.dropout(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        return x  # Shape: (batch, 256)


class AcousticMLP(nn.Module):
    """
    MLP for processing acoustic features
    Input: (batch, num_features)
    Output: feature vector
    """
    def __init__(self, num_features, dropout=0.3):
        super(AcousticMLP, self).__init__()
        
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        
        return x  # Shape: (batch, 32)


class HybridVoiceDetector(nn.Module):
    """
    Hybrid model combining CNN and MLP
    Final classification: AI_GENERATED (0) or HUMAN (1)
    """
    def __init__(self, num_acoustic_features, dropout=0.3):
        super(HybridVoiceDetector, self).__init__()
        
        # Two branches
        self.cnn_branch = SpectrogramCNN(dropout=dropout)
        self.mlp_branch = AcousticMLP(num_acoustic_features, dropout=dropout)
        
        # Fusion layer
        # CNN outputs 256, MLP outputs 32 -> combined = 288
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )
    
    def forward(self, mel_spec, acoustic_features):
        """
        Args:
            mel_spec: (batch, 1, 128, time_steps)
            acoustic_features: (batch, num_features)
        Returns:
            logits: (batch, 2)
        """
        # Extract features from both branches
        cnn_features = self.cnn_branch(mel_spec)
        mlp_features = self.mlp_branch(acoustic_features)
        
        # Concatenate
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        
        # Final classification
        logits = self.fusion(combined)
        
        return logits
    
    def predict(self, mel_spec, acoustic_features):
        """
        Predict with confidence scores
        Returns: (prediction, confidence)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(mel_spec, acoustic_features)
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        return prediction, confidence


# Test the model
if __name__ == "__main__":
    # Test model creation
    print("🧪 Testing Model Architecture...")
    
    # Create dummy data
    batch_size = 4
    mel_spec = torch.randn(batch_size, 1, 128, 157)  # 157 is typical time_steps for 5s audio
    acoustic_features = torch.randn(batch_size, 45)  # 45 features (example)
    
    # Create model
    model = HybridVoiceDetector(num_acoustic_features=45)
    
    # Forward pass
    output = model(mel_spec, acoustic_features)
    
    print(f"✅ Model created successfully!")
    print(f"✅ Input shapes:")
    print(f"   - Mel Spectrogram: {mel_spec.shape}")
    print(f"   - Acoustic Features: {acoustic_features.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output logits: {output[0]}")
    
    # Test prediction
    prediction, confidence = model.predict(mel_spec, acoustic_features)
    print(f"✅ Predictions: {prediction}")
    print(f"✅ Confidence: {confidence}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")