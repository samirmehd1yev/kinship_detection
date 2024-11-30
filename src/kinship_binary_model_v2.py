import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from PIL import Image
import wandb
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score
import timm

class Config:
    def __init__(self):
        # Paths
        self.train_csv = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/train_triplets_enhanced.csv'
        self.val_csv = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/val_triplets_enhanced.csv'
        self.test_csv = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/test_triplets_enhanced.csv'
        self.output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/kin_binary_model_v3'
        
        
        # Model settings
        self.backbone = 'resnet50'
        self.pretrained = True
        self.embedding_dim = 512
        self.dropout_rate = 0.2
        
        # Training settings
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.weight_decay = 5e-4
        self.num_workers = 4
        
        # Loss settings
        self.margin = 0.5
        self.scale = 64.0
        
        # Input settings
        self.image_size = 112
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.setup_directories()
    
    def setup_directories(self):
        self.checkpoint_dir = Path(self.output_dir) / "checkpoints"
        self.log_dir = Path(self.output_dir) / "logs"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

class KinshipDataset(Dataset):
    def __init__(self, csv_path, transform=None, train=True):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            img1 = Image.open(row['Anchor']).convert('RGB')
            img2 = Image.open(row['Positive']).convert('RGB')
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            label = torch.tensor(1.0, dtype=torch.float32)
            
            return {
                'img1': img1,
                'img2': img2,
                'label': label,
                'img1_path': row['Anchor'],
                'img2_path': row['Positive']
            }
        except Exception as e:
            logging.error(f"Error loading images: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

class KinshipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool=''  # Disable global pooling in the backbone
        )
        
        # Get the correct feature dimension
        self.feat_dim = self.backbone.num_features
        
        self.embedding = nn.Sequential(
            nn.Linear(self.feat_dim, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        self.similarity = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def extract_features(self, x):
        # Extract backbone features
        x = self.backbone(x)
        
        # Apply global average pooling
        if len(x.shape) > 2:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            
        # Project to embedding space
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)
    
    def forward(self, img1, img2):
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        diff_feat = torch.abs(feat1 - feat2)
        combined = torch.cat([feat1, feat2, diff_feat], dim=1)
        similarity = self.similarity(combined)
        return similarity

class KinshipLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.bce = nn.BCELoss()
    
    def forward(self, predictions, labels):
        return self.bce(predictions, labels)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.setup_wandb()
        self.setup_data()
        self.setup_model()
        self.setup_training()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=self.config.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def setup_wandb(self):
        wandb.init(
            project="kinship-verification",
            config=vars(self.config)
        )
    
    def setup_data(self):
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])
        
        # Datasets
        self.train_dataset = KinshipDataset(
            self.config.train_csv,
            transform=train_transform,
            train=True
        )
        
        self.val_dataset = KinshipDataset(
            self.config.val_csv,
            transform=val_transform,
            train=False
        )
        
        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def setup_model(self):
        self.model = KinshipModel(self.config).to(self.config.device)
        self.criterion = KinshipLoss(
            margin=self.config.margin,
            scale=self.config.scale
        ).to(self.config.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )
        
        self.scaler = GradScaler()
    
    def setup_training(self):
        self.best_val_auc = 0
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            img1 = batch['img1'].to(self.config.device)
            img2 = batch['img2'].to(self.config.device)
            label = batch['label'].to(self.config.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                pred = self.model(img1, img2)
                loss = self.criterion(pred, label)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            predictions.extend(pred.detach().cpu().numpy())
            labels.extend(label.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(labels, predictions)
        
        return epoch_loss, epoch_auc
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            img1 = batch['img1'].to(self.config.device)
            img2 = batch['img2'].to(self.config.device)
            label = batch['label'].to(self.config.device)
            
            pred = self.model(img1, img2)
            loss = self.criterion(pred, label)
            
            total_loss += loss.item()
            predictions.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())
        
        val_loss = total_loss / len(self.val_loader)
        val_auc = roc_auc_score(labels, predictions)
        
        return val_loss, val_auc
    
    def save_checkpoint(self, epoch, val_auc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_auc': val_auc
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, 
                  self.config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 
                      self.config.checkpoint_dir / 'best_model.pth')
    
    def train(self):
        for epoch in range(self.config.epochs):
            logging.info(f"Epoch {epoch+1}/{self.config.epochs}")
            
            # Train
            train_loss, train_auc = self.train_epoch()
            
            # Validate
            val_loss, val_auc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            
            wandb.log(metrics)
            logging.info(f"Metrics: {metrics}")
            
            # Save checkpoint
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_auc, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logging.info("Early stopping triggered")
                break

def main():
    # Initialize config
    config = Config()
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()
    
    # Finish wandb
    wandb.finish()

if __name__ == "__main__":
    main()