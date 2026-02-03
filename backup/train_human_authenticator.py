"""
Train Human Voice Authenticator
Uses ONLY human voices with contrastive learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ml_models.dataset import VoiceDataset
from ml_models.human_voice_authenticator import HumanVoiceAuthenticator


class HumanAuthenticatorTrainer:
    """Train human voice authenticator using only human samples"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Create save directory
        self.save_dir = Path('saved_models') / config['experiment_name']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        self.setup_data()
        self.setup_model()
        
        # For storing human voice centroid
        self.human_centroid = None
        self.best_val_loss = float('inf')
    
    def setup_data(self):
        """Load datasets"""
        print("\n📂 Loading datasets...")
        
        # Load full dataset
        full_train = VoiceDataset('dataset', split='train', cache_features=True)
        full_val = VoiceDataset('dataset', split='validation', cache_features=True)
        
        # Filter ONLY HUMAN samples (label = 1)
        train_human_indices = [i for i, label in enumerate(full_train.labels) if label == 1]
        val_human_indices = [i for i, label in enumerate(full_val.labels) if label == 1]
        
        print(f"\n📊 Human Voice Statistics:")
        print(f"   Train: {len(train_human_indices)} human samples")
        print(f"   Validation: {len(val_human_indices)} human samples")
        
        # Create subset datasets
        self.train_dataset = torch.utils.data.Subset(full_train, train_human_indices)
        self.val_dataset_human = torch.utils.data.Subset(full_val, val_human_indices)
        
        # Also keep AI samples for validation (to test detection)
        val_ai_indices = [i for i, label in enumerate(full_val.labels) if label == 0]
        self.val_dataset_ai = torch.utils.data.Subset(full_val, val_ai_indices)
        
        print(f"   Validation AI (for testing): {len(val_ai_indices)} samples")
        
        # Get feature count
        num_features = full_train.get_num_acoustic_features()
        print(f"   Acoustic features: {num_features}")
        self.config['num_acoustic_features'] = num_features
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader_human = DataLoader(
            self.val_dataset_human,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        self.val_loader_ai = DataLoader(
            self.val_dataset_ai,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
    
    def setup_model(self):
        """Setup model"""
        print("\n🧠 Creating model...")
        
        self.model = HumanVoiceAuthenticator(
            num_acoustic_features=self.config['num_acoustic_features'],
            dropout=self.config['dropout']
        )
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✅ Model parameters: {total_params:,}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
    
    def compute_human_centroid(self):
        """Compute centroid of human voice embeddings"""
        print("\n🎯 Computing human voice centroid...")
        
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for mel_specs, acoustic_feats, _ in tqdm(self.train_loader, desc="Computing centroid"):
                mel_specs = mel_specs.to(self.device)
                acoustic_feats = acoustic_feats.to(self.device)
                
                embeddings = self.model(mel_specs, acoustic_feats)
                all_embeddings.append(embeddings.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        centroid = torch.mean(all_embeddings, dim=0)
        
        self.human_centroid = centroid.to(self.device)
        print(f"   ✅ Centroid computed: shape {self.human_centroid.shape}")
    
    def train_epoch(self, epoch):
        """Train for one epoch using triplet loss"""
        self.model.train()
        
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for mel_specs, acoustic_feats, _ in pbar:
            mel_specs = mel_specs.to(self.device)
            acoustic_feats = acoustic_feats.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get embeddings
            embeddings = self.model(mel_specs, acoustic_feats)
            
            # Loss: Pull all human voices toward centroid
            centroid_loss = torch.mean(torch.norm(embeddings - self.human_centroid, dim=1))
            
            # Compactness loss: Keep human voices close together
            pairwise_distances = torch.cdist(embeddings, embeddings)
            compactness_loss = torch.mean(pairwise_distances)
            
            # Combined loss
            loss = centroid_loss + 0.1 * compactness_loss
            
            loss.backward()
            self.optimizer.step()
            
            # Update centroid (moving average)
            with torch.no_grad():
                new_centroid = torch.mean(embeddings, dim=0)
                self.human_centroid = 0.9 * self.human_centroid + 0.1 * new_centroid
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate by checking separation between human and AI"""
        self.model.eval()
        
        human_distances = []
        ai_distances = []
        
        # Compute distances for human voices
        with torch.no_grad():
            for mel_specs, acoustic_feats, _ in tqdm(self.val_loader_human, desc=f'Epoch {epoch+1} [Val Human]'):
                mel_specs = mel_specs.to(self.device)
                acoustic_feats = acoustic_feats.to(self.device)
                
                embeddings = self.model(mel_specs, acoustic_feats)
                distances = torch.norm(embeddings - self.human_centroid, dim=1)
                human_distances.extend(distances.cpu().numpy())
        
        # Compute distances for AI voices
        if len(self.val_loader_ai) > 0:
            with torch.no_grad():
                for mel_specs, acoustic_feats, _ in tqdm(self.val_loader_ai, desc=f'Epoch {epoch+1} [Val AI]'):
                    mel_specs = mel_specs.to(self.device)
                    acoustic_feats = acoustic_feats.to(self.device)
                    
                    embeddings = self.model(mel_specs, acoustic_feats)
                    distances = torch.norm(embeddings - self.human_centroid, dim=1)
                    ai_distances.extend(distances.cpu().numpy())
        
        # Statistics
        human_distances = np.array(human_distances)
        ai_distances = np.array(ai_distances) if ai_distances else np.array([])
        
        human_mean = np.mean(human_distances)
        human_std = np.std(human_distances)
        
        print(f"\n   📊 Distance Statistics:")
        print(f"      Human: μ={human_mean:.4f}, σ={human_std:.4f}")
        
        if len(ai_distances) > 0:
            ai_mean = np.mean(ai_distances)
            ai_std = np.std(ai_distances)
            print(f"      AI:    μ={ai_mean:.4f}, σ={ai_std:.4f}")
            print(f"      Separation: {(ai_mean - human_mean):.4f}")
            
            # Find optimal threshold
            threshold = (human_mean + ai_mean) / 2
            
            # Calculate accuracy
            human_correct = np.sum(human_distances < threshold)
            ai_correct = np.sum(ai_distances >= threshold)
            
            human_acc = 100 * human_correct / len(human_distances)
            ai_acc = 100 * ai_correct / len(ai_distances) if len(ai_distances) > 0 else 0
            balanced_acc = (human_acc + ai_acc) / 2
            
            print(f"      Threshold: {threshold:.4f}")
            print(f"      Human Accuracy: {human_acc:.2f}%")
            print(f"      AI Detection: {ai_acc:.2f}%")
            print(f"      Balanced: {balanced_acc:.2f}%")
            
            return human_mean, balanced_acc, threshold
        else:
            return human_mean, 100.0, human_mean + 2 * human_std
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("🚀 TRAINING HUMAN VOICE AUTHENTICATOR")
        print("="*60)
        
        # Initial centroid
        self.compute_human_centroid()
        
        start_time = datetime.now()
        best_accuracy = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']}")
            print("-" * 60)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, accuracy, threshold = self.validate(epoch)
            
            print(f"\n   📈 Train Loss: {train_loss:.4f}")
            print(f"   📉 Val Loss: {val_loss:.4f}")
            print(f"   🎯 Accuracy: {accuracy:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'human_centroid': self.human_centroid,
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'config': self.config
                }, self.save_dir / 'best_model.pth')
                
                print(f"   💾 Saved best model (accuracy: {accuracy:.2f}%)")
            
            # Save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'human_centroid': self.human_centroid,
                'threshold': threshold,
                'accuracy': accuracy,
                'config': self.config
            }, self.save_dir / 'latest_model.pth')
        
        duration = datetime.now() - start_time
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"   Duration: {duration}")
        print(f"   Best Accuracy: {best_accuracy:.2f}%")
        print(f"   Models saved to: {self.save_dir}")


if __name__ == "__main__":
    config = {
        'experiment_name': f'human_authenticator_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'batch_size': 16,
        'epochs': 30,
        'learning_rate': 0.001,
        'dropout': 0.3,
    }
    
    trainer = HumanAuthenticatorTrainer(config)
    trainer.train()