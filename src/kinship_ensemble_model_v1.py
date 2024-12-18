import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score
import cv2
import os
import pandas as pd
from torchvision import transforms
import onnx
from onnx2torch import convert
from torch import optim
import math
from collections import defaultdict
import json
import time
from torch.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class RelationshipSpecificModel(nn.Module):
    def __init__(self, onnx_path, relationship_type):
        super().__init__()
        onnx_model = onnx.load(onnx_path)
        self.backbone = convert(onnx_model)
        self.backbone.requires_grad_(True)
        self.relationship_type = relationship_type
        self.use_gradient_checkpointing = False
    
    def forward_one(self, x):
        if self.use_gradient_checkpointing and self.training:
            feat = checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        else:
            feat = self.backbone(x)
        return F.normalize(feat, p=2, dim=1)
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings1, embeddings2, labels):
        similarity = F.cosine_similarity(embeddings1, embeddings2)
        labels = labels.float()
        loss = labels * torch.pow(1 - similarity, 2) + \
               (1 - labels) * torch.pow(torch.clamp(similarity - self.margin, min=0.0), 2)
        return loss.mean()

class RelationshipSpecificDataset(Dataset):
    def __init__(self, triplets_df, rel_type, transform=None, is_training=True):
        # Filter dataframe for specific relationship type
        self.triplets_df = triplets_df[triplets_df['ptype'] == rel_type].reset_index(drop=True)
        self.is_training = is_training
        self.rel_type = rel_type
        
        # Fix paths
        for col in ['Anchor', 'Positive', 'Negative']:
            self.triplets_df[col] = self.triplets_df[col].str.replace(
                '../data',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
                regex=False
            )
        
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    transforms.RandomErasing(p=0.1)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.triplets_df) * 2
    
    def load_image(self, image_path):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)
                return img
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.1)
    
    def __getitem__(self, idx):
        row_idx = idx // 2
        is_positive = idx % 2 == 0
        
        row = self.triplets_df.iloc[row_idx]
        
        try:
            anchor_img = self.load_image(row['Anchor'])
            
            if is_positive:
                pair_img = self.load_image(row['Positive'])
                label = 1
            else:
                pair_img = self.load_image(row['Negative'])
                label = 0
            
            assert torch.isfinite(anchor_img).all(), f"Found inf/nan in anchor image at idx {idx}"
            assert torch.isfinite(pair_img).all(), f"Found inf/nan in pair image at idx {idx}"
            assert anchor_img.max() <= 1.0 and anchor_img.min() >= -1.0, "Image normalization issue"
            
            return {
                'anchor': anchor_img,
                'pair': pair_img,
                'is_kin': torch.LongTensor([label])
            }
        except Exception as e:
            print(f"Error loading images for row {row_idx}: {str(e)}")
            raise

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EnsembleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.relationship_types = ['ms', 'md', 'fs', 'fd', 'ss', 'bb', 'sibs']
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.criterion = ContrastiveLoss(margin=config['margin'])
        
        # Initialize models for each relationship type
        for rel_type in self.relationship_types:
            self.models[rel_type] = RelationshipSpecificModel(
                config['onnx_path'], 
                rel_type
            ).to(self.device)
            
            self.optimizers[rel_type] = optim.AdamW(
                self.models[rel_type].parameters(),
                lr=config['lr_backbone'],
                weight_decay=config['weight_decay']
            )
            
            # Initialize schedulers
            steps_per_epoch = config['steps_per_epoch']
            total_steps = steps_per_epoch * config['epochs']
            warmup_steps = steps_per_epoch * config['warmup_epochs']
            
            self.schedulers[rel_type] = get_cosine_schedule_with_warmup(
                self.optimizers[rel_type],
                warmup_steps,
                total_steps
            )
    
    def calculate_metrics(self, all_preds, all_labels):
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        
        thresholds = np.arange(-1.0, 1.0, 0.01)
        best_acc = 0
        best_threshold = 0
        
        for threshold in thresholds:
            pred_binary = (all_preds >= threshold).astype(int)
            acc = accuracy_score(all_labels, pred_binary)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        return {
            'accuracy': best_acc,
            'auc': auc,
            'threshold': best_threshold
        }
    
    def train_relationship_specific(self, dataloaders, rel_type):
        best_val_metrics = {
            'accuracy': 0,
            'auc': 0,
            'epoch': 0,
            'model_state': None,
            'threshold': 0
        }
        
        model = self.models[rel_type]
        optimizer = self.optimizers[rel_type]
        scheduler = self.schedulers[rel_type]
        scaler = GradScaler()
        patience_counter = 0
        
        train_loader = dataloaders[rel_type]['train']
        val_loader = dataloaders[rel_type]['val']
        
        try:
            for epoch in range(self.config['epochs']):
                # Training phase
                model.train()
                running_loss = 0
                running_acc = 0
                num_batches = 0
                all_train_preds = []
                all_train_labels = []
                
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} - {rel_type}')
                
                for batch_idx, batch in enumerate(pbar):
                    try:
                        # Get samples
                        anchor = batch['anchor'].to(self.device)
                        pair = batch['pair'].to(self.device)
                        is_kin = batch['is_kin'].to(self.device)
                        
                        # Clear memory if needed
                        if batch_idx % 50 == 0:
                            torch.cuda.empty_cache()
                        
                        with autocast('cuda'):
                            emb1, emb2 = model(anchor, pair)
                            loss = self.criterion(emb1, emb2, is_kin.squeeze())
                        
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        
                        with torch.no_grad():
                            similarity = F.cosine_similarity(emb1, emb2)
                            preds = (similarity > best_val_metrics['threshold']).float()
                            acc = (preds == is_kin.squeeze().float()).float().mean()
                            
                            all_train_preds.extend(similarity.cpu().numpy())
                            all_train_labels.extend(is_kin.squeeze().cpu().numpy())
                        
                        running_loss += loss.item()
                        running_acc += acc.item()
                        num_batches += 1
                        
                        pbar.set_postfix({
                            'loss': f"{running_loss/num_batches:.4f}",
                            'acc': f"{running_acc/num_batches:.4f}",
                            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                        })
                        
                    except RuntimeError as e:
                        print(f"Error in batch {batch_idx}: {str(e)}")
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                        continue
                    
                    except Exception as e:
                        print(f"Unexpected error in batch {batch_idx}: {str(e)}")
                        continue
                
                if num_batches == 0:
                    print(f"No valid batches in epoch {epoch+1}, skipping...")
                    continue
                
                # Calculate epoch metrics
                epoch_loss = running_loss / num_batches
                epoch_acc = running_acc / num_batches
                epoch_auc = roc_auc_score(all_train_labels, all_train_preds)
                
                # Validation phase
                val_metrics = self.evaluate_model(model, val_loader)
                
                print(f'\nEpoch {epoch+1} Summary for {rel_type}:')
                print(f'Training - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, AUC: {epoch_auc:.4f}')
                print(f'Validation - Acc: {val_metrics["accuracy"]:.4f}, AUC: {val_metrics["auc"]:.4f}')
                
                # Log metrics
                wandb.log({
                    f'{rel_type}_train_loss': epoch_loss,
                    f'{rel_type}_train_acc': epoch_acc,
                    f'{rel_type}_train_auc': epoch_auc,
                    f'{rel_type}_val_acc': val_metrics['accuracy'],
                    f'{rel_type}_val_auc': val_metrics['auc'],
                    f'{rel_type}_learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch
                })
                
                # Save best model
                if val_metrics['accuracy'] > best_val_metrics['accuracy']:
                    best_val_metrics.update({
                        'accuracy': val_metrics['accuracy'],
                        'auc': val_metrics['auc'],
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'threshold': val_metrics['threshold']
                    })
                    
                    checkpoint_path = f'checkpoints/kin_binary_ensemble_v1/{rel_type}_best_model.pth'
                    os.makedirs('checkpoints/kin_binary_ensemble_v1', exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_metrics': val_metrics,
                        'config': self.config,
                        'scaler_state_dict': scaler.state_dict()
                    }, checkpoint_path)
                    
                    print(f'Saved new best model for {rel_type} with validation accuracy: {val_metrics["accuracy"]:.4f}')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['patience']:
                        print(f'Early stopping for {rel_type} at epoch {epoch+1}')
                        break
        
        except Exception as e:
            print(f"Training error for {rel_type}: {str(e)}")
            if best_val_metrics['model_state'] is not None:
                print("Returning best model found before error")
                return best_val_metrics
            raise
        
        return best_val_metrics

    def evaluate_model(self, model, data_loader):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                anchor = batch['anchor'].to(self.device)
                pair = batch['pair'].to(self.device)
                is_kin = batch['is_kin'].squeeze()
                
                emb1, emb2 = model(anchor, pair)
                similarity = F.cosine_similarity(emb1, emb2)
                
                all_preds.extend(similarity.cpu().numpy())
                all_labels.extend(is_kin.numpy())
        
        return self.calculate_metrics(all_preds, all_labels)

    def train_all_models(self, dataloaders):
        results = {}
        
        for rel_type in self.relationship_types:
            print(f"\nTraining model for {rel_type} relationship...")
            best_metrics = self.train_relationship_specific(
                dataloaders,
                rel_type
            )
            
            # Test performance
            model = self.models[rel_type]
            model.load_state_dict(best_metrics['model_state'])
            test_loader = dataloaders[rel_type]['test']
            test_metrics = self.evaluate_model(model, test_loader)
            
            results[rel_type] = {
                'val_metrics': best_metrics,
                'test_metrics': test_metrics
            }
            
            print(f"\nResults for {rel_type}:")
            print(f"Validation - Acc: {best_metrics['accuracy']:.4f}, AUC: {best_metrics['auc']:.4f}")
            print(f"Test - Acc: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['auc']:.4f}")
        
        return results

def create_relationship_specific_dataloaders(base_path, batch_size=128):
    """
    Create separate dataloaders for each relationship type
    """
    # Load the split CSVs
    train_df = pd.read_csv(os.path.join(base_path, 'train_triplets_enhanced.csv'))
    val_df = pd.read_csv(os.path.join(base_path, 'val_triplets_enhanced.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test_triplets_enhanced.csv'))
    
    relationship_types = ['ms', 'md', 'fs', 'fd', 'ss', 'bb', 'sibs']
    dataloaders = {}
    
    for rel_type in relationship_types:
        # Create datasets for this relationship type
        train_dataset = RelationshipSpecificDataset(train_df, rel_type, is_training=True)
        val_dataset = RelationshipSpecificDataset(val_df, rel_type, is_training=False)
        test_dataset = RelationshipSpecificDataset(test_df, rel_type, is_training=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        dataloaders[rel_type] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        # Print dataset sizes
        print(f"\nDataset sizes for {rel_type}:")
        print(f"Train: {len(train_dataset)} pairs ({len(train_dataset)//2} triplets)")
        print(f"Validation: {len(val_dataset)} pairs ({len(val_dataset)//2} triplets)")
        print(f"Test: {len(test_dataset)} pairs ({len(test_dataset)//2} triplets)")
    
    return dataloaders

def main():
    # Configuration
    config = {
        'batch_size': 128,
        'lr_backbone': 1e-5,
        'epochs': 15,
        'warmup_epochs': 2,
        'weight_decay': 0.01,
        'patience': 3,
        'margin': 0.3,
        'onnx_path': os.path.join(os.path.expanduser('~'), 
                                 '.insightface/models/buffalo_l/w600k_r50.onnx')
    }
    
    # Initialize wandb
    wandb.init(
        project="kinship-verification-ensemble",
        config=config,
        tags=['ensemble', 'relationship-specific']
    )
    
    # Create datasets and dataloaders
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2'
    dataloaders = create_relationship_specific_dataloaders(base_path, config['batch_size'])
    
    # Add steps_per_epoch to config for each relationship type
    config['steps_per_epoch'] = min([len(dataloaders[rel_type]['train']) 
                                   for rel_type in dataloaders.keys()])
    
    # Train ensemble
    trainer = EnsembleTrainer(config)
    results = trainer.train_all_models(dataloaders)
    
    # Save results
    with open('checkpoints/kin_binary_ensemble_v1/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    wandb.finish()

if __name__ == "__main__":
    main()