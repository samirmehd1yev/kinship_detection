import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
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

# set gpu 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class KinshipVerificationModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        onnx_model = onnx.load(onnx_path)
        self.backbone = convert(onnx_model)
        self.backbone.requires_grad_(True)
        
        self.use_gradient_checkpointing = False
        # if hasattr(self.backbone, 'gradient_checkpointing_enable'):
        #     self.backbone.gradient_checkpointing_enable()
    
    def forward_one(self, x):
        if self.use_gradient_checkpointing and self.training:
            emb = checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        else:
            emb = self.backbone(x)
        return emb
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings1, embeddings2, labels):
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(embeddings1, embeddings2)
        
        # Convert labels to float
        labels = labels.float()
        
        # Contrastive loss calculation
        loss = labels * torch.pow(1 - similarity, 2) + \
               (1 - labels) * torch.pow(torch.clamp(similarity - self.margin, min=0.0), 2)
        
        return loss.mean()

class KinshipDataset(Dataset):
    def __init__(self, triplets_df, transform=None, is_training=True):
        self.triplets_df = triplets_df
        self.is_training = is_training
        
        for col in ['Anchor', 'Positive', 'Negative']:
            self.triplets_df[col] = self.triplets_df[col].str.replace(
                'data',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
                regex=False
            )
        
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced intensity
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    transforms.RandomErasing(p=0.1)  # Reduced probability
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
            
            # Add sanity checks
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

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            is_kin = batch['is_kin'].squeeze()
            
            emb1, emb2 = model(anchor, pair)
            similarity = F.cosine_similarity(
                F.normalize(emb1, p=2, dim=1),
                F.normalize(emb2, p=2, dim=1)
            )
            
            all_preds.extend(similarity.cpu().numpy())
            all_labels.extend(is_kin.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_preds)
    
    # Find best threshold using accuracy
    thresholds = np.arange(-1.0, 1.0, 0.01)
    best_acc = 0
    best_threshold = 0
    
    for threshold in thresholds:
        pred_binary = (all_preds >= threshold).astype(int)
        acc = accuracy_score(all_labels, pred_binary)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    # Calculate final metrics with best threshold
    final_preds = (all_preds >= best_threshold).astype(int)
    
    return {
        'auc': auc,
        'accuracy': best_acc,
        'threshold': best_threshold
    }
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(model, train_loader, val_loader, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    scaler = GradScaler('cuda')
    criterion = ContrastiveLoss(margin=config['margin']).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr_backbone'],
        weight_decay=config['weight_decay']
    )

    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = len(train_loader) * config['warmup_epochs']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Evaluate model before training
    print("\nEvaluating model before training:")
    model.eval()
    initial_val_metrics = evaluate_model(model, val_loader, device)
    initial_test_metrics = evaluate_model(model, test_loader, device)
    
    print("\nInitial Performance:")
    print(f"Validation - Accuracy: {initial_val_metrics['accuracy']:.4f}, AUC: {initial_val_metrics['auc']:.4f}")
    print(f"Test - Accuracy: {initial_test_metrics['accuracy']:.4f}, AUC: {initial_test_metrics['auc']:.4f}")
    print(f"Initial best threshold: {initial_val_metrics['threshold']:.4f}")
    
    wandb.log({
        'initial_val_acc': initial_val_metrics['accuracy'],
        'initial_val_auc': initial_val_metrics['auc'],
        'initial_test_acc': initial_test_metrics['accuracy'],
        'initial_test_auc': initial_test_metrics['auc'],
        'initial_threshold': initial_val_metrics['threshold']
    })

    best_val_metrics = {
        'accuracy': initial_val_metrics['accuracy'],
        'auc': initial_val_metrics['auc'],
        'epoch': 0,
        'model_state': model.state_dict(),
        'optimizer_state': None,
        'threshold': initial_val_metrics['threshold']
    }
    
    threshold_metrics = {
        'threshold': initial_val_metrics['threshold'],
        'accuracy': initial_val_metrics['accuracy']
    }
    
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0
        running_acc = 0
        all_train_preds = []
        all_train_labels = []
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            is_kin = batch['is_kin'].to(device)
            
            with autocast('cuda'):
                emb1, emb2 = model(anchor, pair)
                loss = criterion(emb1, emb2, is_kin.squeeze())
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                similarity = F.cosine_similarity(
                    F.normalize(emb1, p=2, dim=1),
                    F.normalize(emb2, p=2, dim=1)
                )
                # Use current best threshold for accuracy calculation
                preds = (similarity > threshold_metrics['threshold']).float()
                acc = (preds == is_kin.squeeze().float()).float().mean()
                
                # Store predictions and labels for AUC calculation
                all_train_preds.extend(similarity.cpu().numpy())
                all_train_labels.extend(is_kin.squeeze().cpu().numpy())
            
            running_loss += loss.item()
            running_acc += acc.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/num_batches:.4f}",
                'acc': f"{running_acc/num_batches:.4f}",
                'threshold': f"{threshold_metrics['threshold']:.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb less frequently
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_acc': acc.item(),
                    'current_threshold': threshold_metrics['threshold'],
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
                
                # Optional: cleanup
                if batch_idx % 500 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_acc = running_acc / num_batches
        epoch_auc = roc_auc_score(all_train_labels, all_train_preds)
        
        scheduler.step()
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, AUC: {epoch_auc:.4f}')
        print(f'Validation - Acc: {val_metrics["accuracy"]:.4f}, AUC: {val_metrics["auc"]:.4f}')
        print(f'Current threshold: {threshold_metrics["threshold"]:.4f}, New threshold: {val_metrics["threshold"]:.4f}')
        
        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'train_auc': epoch_auc,
            'val_acc': val_metrics['accuracy'],
            'val_auc': val_metrics['auc'],
            'best_threshold': val_metrics['threshold']
        })
        
        # Update threshold if accuracy improves
        if val_metrics['accuracy'] > threshold_metrics['accuracy']:
            threshold_metrics['threshold'] = val_metrics['threshold']
            threshold_metrics['accuracy'] = val_metrics['accuracy']
            print(f"New best threshold: {val_metrics['threshold']:.4f} with accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save best model based on accuracy
        if val_metrics['accuracy'] > best_val_metrics['accuracy']:
            best_val_metrics['accuracy'] = val_metrics['accuracy']
            best_val_metrics['auc'] = val_metrics['auc']
            best_val_metrics['epoch'] = epoch
            best_val_metrics['model_state'] = model.state_dict()
            best_val_metrics['optimizer_state'] = optimizer.state_dict()
            best_val_metrics['threshold'] = val_metrics['threshold']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'model_version': '1.0',
                'timestamp': time.strftime('%Y%m%d-%H%M%S'),
                'scaler_state_dict': scaler.state_dict(),
                'best_threshold': val_metrics['threshold']
            }
            
            torch.save(checkpoint, 'checkpoints/kin_binary_v1/best_model_l.pth')
            print(f'Saved new best model with validation accuracy: {val_metrics["accuracy"]:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    return best_val_metrics
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_training', action='store_true', help='Continue training from last checkpoint')
    parser.add_argument('--test_only', action='store_true', help='Run only testing on best model')
    args = parser.parse_args()

    config = {
        'batch_size': 128,
        'lr_backbone': 1e-5,
        'epochs': 15,
        'weight_decay': 0.01,
        'patience': 7,
        'warmup_epochs': 2,
        'margin': 0.5
    }

    # Create checkpoints directory
    os.makedirs('checkpoints/kin_binary_v1', exist_ok=True)

    # Get ONNX model path
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    
    # Initialize model
    model = KinshipVerificationModel(onnx_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load data splits
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand'
    
    train_data = pd.read_csv(os.path.join(base_path, 'train_triplets_enhanced.csv'))
    val_data = pd.read_csv(os.path.join(base_path, 'val_triplets_enhanced.csv'))
    test_data = pd.read_csv(os.path.join(base_path, 'test_triplets_enhanced.csv'))
    
    # Print dataset sizes and verify balance
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_data)} triplets ({len(train_data) * 2} pairs - balanced)")
    print(f"Val: {len(val_data)} triplets ({len(val_data) * 2} pairs - balanced)")
    print(f"Test: {len(test_data)} triplets ({len(test_data) * 2} pairs - balanced)")
    
    # Create datasets and loaders
    train_dataset = KinshipDataset(train_data, is_training=True)
    val_dataset = KinshipDataset(val_data, is_training=False)
    test_dataset = KinshipDataset(test_data, is_training=False)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    }

    if args.test_only:
        print("\nRunning comprehensive testing on best model...")
        best_checkpoint = torch.load('checkpoints/kin_binary_v1/best_model_l.pth')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()

        results = {}
        for split in ['val', 'test']:
            metrics = evaluate_model(model, dataloaders[split], device)
            results[split] = metrics
            print(f"\n{split.capitalize()} Set Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"AUC: {metrics['auc']:.4f}")
            print(f"Best Threshold: {metrics['threshold']:.4f}")

        comprehensive_results = {
            'metrics': results,
            'config': config,
            'model_info': {
                'checkpoint_path': 'checkpoints/kin_binary_v1/best_model_l.pth',
                'epoch': best_checkpoint['epoch'],
                'timestamp': time.strftime('%Y%m%d-%H%M%S')
            }
        }
        
        with open('checkpoints/kin_binary_v1/comprehensive_test_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=4)
        
        print("\nComprehensive results saved to: checkpoints/kin_binary_v1/comprehensive_test_results.json")
        return

    # Initialize wandb for training
    run = wandb.init(
        project="kinship-verification-contrastive",
        config=config,
        tags=['contrastive', 'accuracy_metric']
    )

    if args.continue_training:
        print("\nContinuing training from last checkpoint...")
        checkpoint = torch.load('checkpoints/kin_binary_v1/best_model_l.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = optim.AdamW(model.parameters(), lr=config['lr_backbone'], weight_decay=config['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        steps = len(dataloaders['train']) * config['epochs']
        warmup_steps = len(dataloaders['train']) * config['warmup_epochs']
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed from epoch {checkpoint['epoch']} with metrics:")
        print(f"Accuracy: {checkpoint['val_metrics']['accuracy']:.4f}")
        print(f"AUC: {checkpoint['val_metrics']['auc']:.4f}")
        print(f"Best Threshold: {checkpoint['val_metrics']['threshold']:.4f}")
        
        config.update(checkpoint['config'])
    
    # Train model
    best_val_metrics = train_model(model, dataloaders['train'], dataloaders['val'], dataloaders['test'], config)
    print(f'\nTraining completed! Best validation metrics:')
    print(f"Accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"AUC: {best_val_metrics['auc']:.4f}")
    print(f"At epoch: {best_val_metrics['epoch']+1}")
    
    # Final evaluation on test set
    model.load_state_dict(best_val_metrics['model_state'])
    test_metrics = evaluate_model(model, dataloaders['test'], device)
    
    print('\nFinal Test Set Results:')
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"Best Threshold: {test_metrics['threshold']:.4f}")
    
    final_results = {
        'best_val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'config': config,
        'timestamp': time.strftime('%Y%m%d-%H%M%S')
    }
    
    with open('checkpoints/kin_binary_v1/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    wandb.finish()

if __name__ == "__main__":
    main()