"""
Training Script for AI Voice Detection
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

# Add ml_models to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_models.dataset import VoiceDataset
from ml_models.model import HybridVoiceDetector

class Trainer:
    """Training pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Create save directory
        self.save_dir = Path('saved_models') / config['experiment_name']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
        # Training metrics
        self.best_val_acc = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        print("\n📂 Loading datasets...")
        
        # Load datasets
        self.train_dataset = VoiceDataset(
            'dataset',
            split='train',
            cache_features=True
        )
        
        self.val_dataset = VoiceDataset(
            'dataset',
            split='validation',
            cache_features=True
        )
        
        # Get number of acoustic features
        num_features = self.train_dataset.get_num_acoustic_features()
        print(f"   Number of acoustic features: {num_features}")
        self.config['num_acoustic_features'] = num_features
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"   ✅ Train samples: {len(self.train_dataset)}")
        print(f"   ✅ Validation samples: {len(self.val_dataset)}")
    
    def setup_model(self):
        """Setup model"""
        print("\n🧠 Creating model...")
        
        self.model = HybridVoiceDetector(
            num_acoustic_features=self.config['num_acoustic_features'],
            dropout=self.config['dropout']
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✅ Model parameters: {total_params:,}")
    
    def setup_training(self):
        """Setup loss, optimizer, scheduler"""
        print("\n⚙️  Setting up training...")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        print(f"   ✅ Optimizer: Adam (lr={self.config['learning_rate']})")
        print(f"   ✅ Scheduler: ReduceLROnPlateau")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for mel_specs, acoustic_feats, labels in pbar:
            # Move to device
            mel_specs = mel_specs.to(self.device)
            acoustic_feats = acoustic_feats.to(self.device)
            labels = labels.to(self.device).squeeze()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(mel_specs, acoustic_feats)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # For detailed metrics
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]  ')
            
            for mel_specs, acoustic_feats, labels in pbar:
                mel_specs = mel_specs.to(self.device)
                acoustic_feats = acoustic_feats.to(self.device)
                labels = labels.to(self.device).squeeze()
                
                outputs = self.model(mel_specs, acoustic_feats)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calculate per-class accuracy
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)
        
        ai_mask = all_labels == 0
        human_mask = all_labels == 1
        
        ai_acc = (all_predictions[ai_mask] == all_labels[ai_mask]).float().mean() * 100
        human_acc = (all_predictions[human_mask] == all_labels[human_mask]).float().mean() * 100
        
        print(f'   AI Detection Accuracy: {ai_acc:.2f}%')
        print(f'   Human Detection Accuracy: {human_acc:.2f}%')
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest_model.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            print(f'   💾 Saved best model (val_acc: {self.best_val_acc:.2f}%)')
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("🚀 TRAINING STARTED")
        print("="*60)
        
        start_time = datetime.now()
        
        for epoch in range(self.config['epochs']):
            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print epoch summary
            print(f"\n   📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, is_best)
            
            # Save training history
            history = {
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs
            }
            
            with open(self.save_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
        
        # Training complete
        duration = datetime.now() - start_time
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print("="*60)
        print(f"   Duration: {duration}")
        print(f"   Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"   Models saved to: {self.save_dir}")


if __name__ == "__main__":
    # Training configuration
    config = {
        'experiment_name': f'voice_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'batch_size': 16,  # Reduced for your GPU
        'epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'dropout': 0.3,
    }
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()