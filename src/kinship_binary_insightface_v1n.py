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
import random
from torchsummary import summary

# set gpu 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # scale
        self.m = m  # margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class KinshipVerificationModel(nn.Module):
    def __init__(self, onnx_path, num_classes=2):
        super().__init__()
        onnx_model = onnx.load(onnx_path)
        self.backbone = convert(onnx_model)
        self.backbone.requires_grad_(True)
        
        self.embedding_size = 512
        self.bn = nn.BatchNorm1d(self.embedding_size)  # add BN
        self.dropout = nn.Dropout(0.2)  # add dropout
        
        self.arc_margin = ArcMarginProduct(
            in_features=self.embedding_size,
            out_features=num_classes,
            s=30.0,
            m=0.5
        )

    def forward(self, x1, x2, labels=None):
        # Extract features
        emb1 = self.backbone(x1)
        emb2 = self.backbone(x2)
        
        # Apply BN and dropout
        emb1 = self.dropout(self.bn(emb1))
        emb2 = self.dropout(self.bn(emb2))
        
        if self.training and labels is not None:
            logits1 = self.arc_margin(emb1, labels)
            logits2 = self.arc_margin(emb2, labels)
            return logits1, logits2
        else:
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)
            return emb1, emb2

class KinshipDataset(Dataset):
    def __init__(self, triplets_df, transform=None, is_training=True):
        self.triplets_df = triplets_df
        self.is_training = is_training
        
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
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
            similarity = F.cosine_similarity(emb1, emb2)
            
            all_preds.extend(similarity.cpu().numpy())
            all_labels.extend(is_kin.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_preds)
    
    # Find best threshold
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
        'auc': auc,
        'accuracy': best_acc,
        'threshold': best_threshold
    }

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model, train_loader, val_loader, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("-" * 50)
    print("\nModel Summary:")
    summary(model, [(3, 112, 112), (3, 112, 112)])
    print("-" * 50)
    scaler = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = len(train_loader) * config['warmup_epochs']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initial evaluation
    print("\nEvaluating model before training:")
    initial_val_metrics = evaluate_model(model, val_loader, device)
    initial_test_metrics = evaluate_model(model, test_loader, device)
    
    print("\nInitial Performance:")
    print(f"Validation - Accuracy: {initial_val_metrics['accuracy']:.4f}, AUC: {initial_val_metrics['auc']:.4f}, Threshold: {initial_val_metrics['threshold']:.4f}")
    print(f"Test - Accuracy: {initial_test_metrics['accuracy']:.4f}, AUC: {initial_test_metrics['auc']:.4f}, Threshold: {initial_test_metrics['threshold']:.4f}")
    
    wandb.log({
        'initial_val_acc': initial_val_metrics['accuracy'],
        'initial_val_auc': initial_val_metrics['auc'],
        'initial_val_threshold': initial_val_metrics['threshold'],
        'initial_test_acc': initial_test_metrics['accuracy'],
        'initial_test_auc': initial_test_metrics['auc'],
        'initial_test_threshold': initial_test_metrics['threshold']
    })
    
    best_val_metrics = {
        'accuracy': initial_val_metrics['accuracy'],
        'auc': initial_val_metrics['auc'],
        'epoch': 0,
        'model_state': model.state_dict(),
        'threshold': initial_val_metrics['threshold']
    }
    
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0
        running_acc = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            is_kin = batch['is_kin'].to(device).squeeze()
            
            with autocast('cuda'):
                logits1, logits2 = model(anchor, pair, is_kin)
                loss1 = criterion(logits1, is_kin)
                loss2 = criterion(logits2, is_kin)
                loss = (loss1 + loss2) / 2
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                # Get embeddings for accuracy calculation
                emb1, emb2 = model(anchor, pair)
                similarity = F.cosine_similarity(emb1, emb2)
                
                # Use current best threshold for accuracy calculation
                predictions = (similarity > best_val_metrics['threshold']).float()
                acc = (predictions == is_kin.float()).float().mean()
            
            running_loss += loss.item()
            running_acc += acc.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{running_loss/num_batches:.4f}",
                'acc': f"{running_acc/num_batches:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_acc': acc.item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        scheduler.step()
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training - Loss: {running_loss/num_batches:.4f}, Acc: {running_acc/num_batches:.4f}, Threshold: {val_metrics["threshold"]:.4f}')
        print(f'Validation - Acc: {val_metrics["accuracy"]:.4f}, AUC: {val_metrics["auc"]:.4f}, Threshold: {val_metrics["threshold"]:.4f}')
        
        wandb.log({
            'epoch': epoch,
            'train_loss': running_loss/num_batches,
            'train_acc': running_acc/num_batches,
            'val_acc': val_metrics['accuracy'],
            'val_auc': val_metrics['auc'],
            'val_threshold': val_metrics['threshold'],
        })
        
        if val_metrics['accuracy'] > best_val_metrics['accuracy']:
            best_val_metrics.update({
                'accuracy': val_metrics['accuracy'],
                'auc': val_metrics['auc'],
                'epoch': epoch,
                'model_state': model.state_dict(),
                'threshold': val_metrics['threshold']
            })
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'model_version': '1.0',
                'timestamp': time.strftime('%Y%m%d-%H%M%S'),
                'scaler_state_dict': scaler.state_dict()
            }, 'checkpoints/kin_binary_v1n/best_model.pth')
            
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
        'lr': 1e-4,  
        'epochs': 15,
        'weight_decay': 1e-2,
        'patience': 7,  
        'warmup_epochs': 2,
        'num_classes': 2
    }

    # Create checkpoints directory
    os.makedirs('checkpoints/kin_binary_v1n', exist_ok=True)

    # Get ONNX model path
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    
    # Initialize model
    model = KinshipVerificationModel(onnx_path, num_classes=config['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load data splits
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand'
    
    train_data = pd.read_csv(os.path.join(base_path, 'train_triplets_enhanced.csv'))
    val_data = pd.read_csv(os.path.join(base_path, 'val_triplets_enhanced.csv'))
    test_data = pd.read_csv(os.path.join(base_path, 'test_triplets_enhanced.csv'))
    
    # Print dataset sizes
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

    def convert_tensors_to_python(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().item() if obj.numel() == 1 else obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {key: convert_tensors_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors_to_python(item) for item in obj]
        return obj

    if args.test_only:
        print("\nRunning comprehensive testing on best model...")
        best_checkpoint = torch.load('checkpoints/kin_binary_v1n/best_model.pth')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()

        results = {}
        for split in ['val', 'test']:
            metrics = evaluate_model(model, dataloaders[split], device)
            results[split] = convert_tensors_to_python(metrics)
            print(f"\n{split.capitalize()} Set Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"AUC: {metrics['auc']:.4f}")
            print(f"Best Threshold: {metrics['threshold']:.4f}")

        comprehensive_results = {
            'metrics': results,
            'config': config,
            'model_info': {
                'checkpoint_path': 'checkpoints/kin_binary_v1n/best_model.pth',
                'epoch': best_checkpoint['epoch'],
                'timestamp': time.strftime('%Y%m%d-%H%M%S')
            }
        }
        
        with open('checkpoints/kin_binary_v1n/comprehensive_test_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=4)
        
        print("\nComprehensive results saved to: checkpoints/kin_binary_v1n/comprehensive_test_results.json")
        return

    # Initialize wandb for training
    run = wandb.init(
        project="kinship-verification-arcface",
        config=config,
        tags=['arcface', 'kinship']
    )

    if args.continue_training:
        print("\nContinuing training from last checkpoint...")
        checkpoint = torch.load('checkpoints/kin_binary_v1n/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        steps = len(dataloaders['train']) * config['epochs']
        warmup_steps = len(dataloaders['train']) * config['warmup_epochs']
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed from epoch {checkpoint['epoch']} with metrics:")
        print(f"Accuracy: {checkpoint['val_metrics']['accuracy']:.4f}")
        print(f"AUC: {checkpoint['val_metrics']['auc']:.4f}")
        
        config.update(checkpoint['config'])
    
    # Train model
    best_val_metrics = train_model(model, dataloaders['train'], dataloaders['val'], dataloaders['test'], config)
    
    # Convert tensor values to Python types
    best_val_metrics = convert_tensors_to_python(best_val_metrics)
    
    print(f'\nTraining completed! Best validation metrics:')
    print(f"Accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"AUC: {best_val_metrics['auc']:.4f}")
    print(f"At epoch: {best_val_metrics['epoch']}")
    
    # Final evaluation on test set
    model.load_state_dict(best_val_metrics['model_state'])
    test_metrics = evaluate_model(model, dataloaders['test'], device)
    test_metrics = convert_tensors_to_python(test_metrics)
    
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
    
    with open('checkpoints/kin_binary_v1n/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    wandb.finish()

if __name__ == "__main__":
    main()