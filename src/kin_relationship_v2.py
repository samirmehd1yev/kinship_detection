import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import json
from torchvision import transforms
import onnx
from onnx2torch import convert
import wandb
from tqdm import tqdm
import numpy as np
import os
import functools
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.metrics import f1_score



# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print = functools.partial(print, flush=True)

# Set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define relationship types and mappings
RELATIONSHIP_TYPES = ['ms', 'md', 'fs', 'fd', 'ss', 'bb', 'sibs']
REL_TO_IDX = {rel: idx for idx, rel in enumerate(RELATIONSHIP_TYPES)}
IDX_TO_REL = {idx: rel for rel, idx in REL_TO_IDX.items()}

# Gender-based relationship constraints
VALID_RELATIONSHIPS = {
    (0, 0): ['md', 'ss'],      # Female-Female
    (1, 1): ['fs', 'bb'],      # Male-Male
    (0, 1): ['ms', 'fd', 'sibs'],  # Female-Male
    (1, 0): ['ms', 'fd', 'sibs']   # Male-Female
}

class ImprovedGenderAwareLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        # Adjust weights to address class imbalance
        if class_weights is None:
            class_weights = torch.tensor([
                0.8,   # ms - well performing
                0.8,   # md - well performing
                0.8,   # fs - well performing
                0.8,   # fd - well performing
                1.2,   # ss - needs boost
                1.2,   # bb - needs boost
                1.2    # sibs - needs boost
            ])
        self.class_weights = class_weights
        self.gamma = 2.0  # focusing parameter
        self.alpha = 0.25  # balancing parameter
        
    def forward(self, logits, labels, gender_features, features=None):
        # Compute focal loss
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        # Apply class weights
        weights = self.class_weights.to(logits.device)[labels]
        weighted_loss = weights * focal_loss
        
        # Enhanced gender constraint loss with smooth penalties
        gender1, gender2 = gender_features[:, 0], gender_features[:, 1]
        gender_loss = torch.zeros_like(ce_loss)
        
        pred_probs = F.softmax(logits, dim=1)
        for i in range(len(labels)):
            valid_rels = VALID_RELATIONSHIPS[(int(gender1[i].item()), int(gender2[i].item()))]
            valid_indices = torch.tensor([REL_TO_IDX[rel] for rel in valid_rels], 
                                       device=logits.device)
            
            # Smooth penalty based on probability mass assigned to invalid relationships
            invalid_probs = pred_probs[i][~torch.isin(torch.arange(len(pred_probs[i])).to(logits.device), 
                                                     valid_indices)]
            gender_loss[i] = invalid_probs.sum() * 2.0
        
        # Contrastive learning component
        contrastive_loss = 0
        if features is not None:
            # Compute similarity matrix
            features = F.normalize(features, dim=1)
            sim_matrix = torch.matmul(features, features.t())
            
            # Create masks for positive and negative pairs
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            gender_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
            
            for i in range(len(labels)):
                for j in range(len(labels)):
                    g1, g2 = gender_features[i], gender_features[j]
                    valid_rels = VALID_RELATIONSHIPS[(int(g1[0].item()), int(g1[1].item()))]
                    if IDX_TO_REL[labels[j].item()] in valid_rels:
                        gender_mask[i, j] = True
            
            valid_mask = pos_mask & gender_mask
            
            # InfoNCE loss computation
            temperature = 0.07
            logits_mask = torch.ones_like(sim_matrix)
            logits_mask.fill_diagonal_(0)
            exp_logits = torch.exp(sim_matrix / temperature) * logits_mask
            log_prob = sim_matrix / temperature - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = -(log_prob * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
            contrastive_loss = mean_log_prob_pos.mean()
        
        # Combine losses with adaptive weighting
        total_loss = weighted_loss + gender_loss + 0.1 * contrastive_loss
        
        return total_loss.mean()

class MultiScaleAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, input_dim)
        )
        
    def forward(self, x):
        # Self attention block
        attended, _ = self.attention(x, x, x)
        x = x + attended
        x = self.norm1(x)
        
        # Feed forward block
        ff_output = self.ffn(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x

class ImprovedRelationshipClassifier(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        print(f"Loading ONNX model from: {onnx_path}")
        self.backbone = convert(onnx.load(onnx_path))
        print("ONNX model loaded successfully")
        
        self.embedding_dim = 512
        self.gender_dim = 2
        self.num_classes = len(RELATIONSHIP_TYPES)
        
        # Enhanced feature extraction
        self.input_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + self.gender_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Multi-scale attention blocks
        self.attention_blocks = nn.ModuleList([
            MultiScaleAttention(1024)
            for _ in range(3)
        ])
        
        # Feature pyramid
        self.feature_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for dim in [512, 256, 128]
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x):
        return self.backbone(x)
    
    def forward(self, x1, x2, gender_features):
        # Extract and normalize embeddings
        emb1 = F.normalize(self.forward_features(x1), p=2, dim=1)
        emb2 = F.normalize(self.forward_features(x2), p=2, dim=1)
        
        # Combine features
        combined = torch.cat([emb1, emb2, gender_features], dim=1)
        features = self.input_proj(combined)
        
        # Apply attention blocks
        features = features.unsqueeze(0)
        for attention in self.attention_blocks:
            features = attention(features)
        features = features.squeeze(0)
        
        # Multi-scale feature extraction
        pyramid_features = []
        for pyramid_layer in self.feature_pyramid:
            pyramid_features.append(pyramid_layer(features))
        
        # Combine multi-scale features
        multi_scale_features = torch.cat(pyramid_features, dim=1)
        
        # Classification with gender constraints
        logits = self.classifier(multi_scale_features)
        
        # Apply gender-based masking
        gender1, gender2 = gender_features[:, 0], gender_features[:, 1]
        mask = torch.full_like(logits, float('-inf'))
        
        for i in range(len(gender1)):
            valid_rels = VALID_RELATIONSHIPS[(int(gender1[i].item()), int(gender2[i].item()))]
            valid_indices = [REL_TO_IDX[rel] for rel in valid_rels]
            mask[i, valid_indices] = 0
        
        return logits + mask, multi_scale_features

class ImprovedRelationshipDataset(Dataset):
    def __init__(self, df, gender_dict, transform=None, is_training=True):
        self.df = df
        self.gender_dict = gender_dict
        self.is_training = is_training
        
        # Update paths
        for col in ['Anchor', 'Positive', 'Negative']:
            self.df[col] = self.df[col].str.replace(
                'data/',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/',
                regex=False
            )
        
        # Enhanced augmentations
        if transform is None:
            if is_training:
                self.transform = A.Compose([
                    A.RandomResizedCrop(112, 112, scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.OneOf([
                        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    ], p=0.5),
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 50.0)),
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.ISONoise(),
                    ], p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(112, 112),
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform
        
        self.load_errors = []
    
    def load_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            
            return img
        except Exception as e:
            self.load_errors.append({
                'path': image_path,
                'error': str(e)
            })
            raise e
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Load images with error handling
            try:
                anchor_img = self.load_image(row['Anchor'])
                positive_img = self.load_image(row['Positive'])
            except Exception as e:
                print(f"Error loading images at index {idx}: {str(e)}")
                if idx + 1 < len(self):
                    return self.__getitem__(idx + 1)
                else:
                    raise RuntimeError("No valid samples found")
            
            # Get relationship label
            rel_type = row['ptype']
            label = torch.tensor(REL_TO_IDX[rel_type])
            
            # Get gender features with validation
            anchor_gender = self.gender_dict.get(row['Anchor'], 0)
            positive_gender = self.gender_dict.get(row['Positive'], 0)
            
            gender_features = torch.tensor([anchor_gender, positive_gender], dtype=torch.float32)
            
            return {
                'anchor': anchor_img,
                'positive': positive_img,
                'gender_features': gender_features,
                'label': label,
                'metadata': {
                    'rel_type': rel_type,
                    'anchor_path': row['Anchor'],
                    'positive_path': row['Positive']
                }
            }
        except Exception as e:
            print(f"Error processing sample at index {idx}: {str(e)}")
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            raise RuntimeError("No valid samples found in the dataset")

def train_model_improved(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize loss function and optimizer
    criterion = ImprovedGenderAwareLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda')
    
    # Setup learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['epochs'] // 3,  # Restart every third of total epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=config['learning_rate'] * 0.01  # Minimum learning rate
    )
    
    best_val_metrics = {
        'accuracy': 0,
        'f1': 0,
        'loss': float('inf')
    }
    patience = 0
    max_patience = 5
    
    # Initialize WandB for experiment tracking
    if config.get('use_wandb', False):
        wandb.init(
            project="kinship-recognition-improved",
            config=config,
            name=config.get('run_name', 'improved_model')
        )
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        train_metrics = {
            'loss': 0,
            'correct': 0,
            'total': 0,
            'gender_violations': 0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            try:
                if len(batch['anchor']) <= 1:
                    continue
                
                metadata = batch.pop('metadata')
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Mixed precision training
                with autocast('cuda'):
                    logits, features = model(batch['anchor'], batch['positive'], batch['gender_features'])
                    loss = criterion(logits, batch['label'], batch['gender_features'], features)
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                with torch.no_grad():
                    pred = logits.argmax(dim=1)
                    train_metrics['correct'] += (pred == batch['label']).sum().item()
                    train_metrics['total'] += len(batch['label'])
                    train_metrics['loss'] += loss.item()
                    
                    # Check gender violations
                    for i in range(len(pred)):
                        gender1, gender2 = batch['gender_features'][i]
                        valid_rels = VALID_RELATIONSHIPS[(int(gender1.item()), int(gender2.item()))]
                        pred_rel = IDX_TO_REL[pred[i].item()]
                        if pred_rel not in valid_rels:
                            train_metrics['gender_violations'] += 1
                
                # Update progress bar
                train_acc = train_metrics['correct'] / train_metrics['total']
                gender_violation_rate = train_metrics['gender_violations'] / train_metrics['total']
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{train_acc:.4f}",
                    'viol': f"{gender_violation_rate:.4f}"
                })
                
                # Log to WandB
                if config.get('use_wandb', False) and batch_idx % 100 == 0:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/accuracy': train_acc,
                        'train/gender_violation_rate': gender_violation_rate,
                        'train/learning_rate': scheduler.get_last_lr()[0]
                    })
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    print(f"CUDA OOM in batch {batch_idx}. Skipping...")
                    continue
                else:
                    raise e
        
        # Validation phase
        val_metrics = evaluate_model_improved(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train - Loss: {train_metrics['loss']/len(train_loader):.4f}, "
              f"Acc: {train_metrics['correct']/train_metrics['total']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Log validation metrics to WandB
        if config.get('use_wandb', False):
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/f1': val_metrics['f1'],
                'val/gender_violation_rate': val_metrics['gender_violation_rate']
            })
        
        # Model checkpointing
        improved = False
        if val_metrics['accuracy'] > best_val_metrics['accuracy']:
            improved = True
            best_val_metrics['accuracy'] = val_metrics['accuracy']
        if val_metrics['f1'] > best_val_metrics['f1']:
            improved = True
            best_val_metrics['f1'] = val_metrics['f1']
        if val_metrics['loss'] < best_val_metrics['loss']:
            improved = True
            best_val_metrics['loss'] = val_metrics['loss']
        
        if improved:
            try:
                save_dir = Path('checkpoints/kin_relationship_v2')
                save_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_metrics': best_val_metrics,
                    'config': config
                }, save_dir / 'best_model.pth')
                
                patience = 0
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping triggered")
                break
    
    return best_val_metrics

def evaluate_model_improved(model, loader, criterion, device, mode='val'):
    model.eval()
    metrics = {
        'correct': 0,
        'total': 0,
        'loss': 0,
        'gender_violations': 0,
        'per_class_correct': {rel: 0 for rel in RELATIONSHIP_TYPES},
        'per_class_total': {rel: 0 for rel in RELATIONSHIP_TYPES},
        'confusion_matrix': np.zeros((len(RELATIONSHIP_TYPES), len(RELATIONSHIP_TYPES))),
        'predictions': [],
        'true_labels': [],
        'errors': []
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating ({mode})"):
            metadata = batch.pop('metadata')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            logits, features = model(batch['anchor'], batch['positive'], batch['gender_features'])
            loss = criterion(logits, batch['label'], batch['gender_features'], features)
            
            pred = logits.argmax(dim=1)
            metrics['correct'] += (pred == batch['label']).sum().item()
            metrics['total'] += len(batch['label'])
            metrics['loss'] += loss.item()
            
            metrics['predictions'].extend(pred.cpu().numpy())
            metrics['true_labels'].extend(batch['label'].cpu().numpy())
            
            # Detailed metrics
            for i, (p, t) in enumerate(zip(pred, batch['label'])):
                # Update confusion matrix
                metrics['confusion_matrix'][t.item()][p.item()] += 1
                
                # Per-class accuracy
                true_rel = IDX_TO_REL[t.item()]
                pred_rel = IDX_TO_REL[p.item()]
                metrics['per_class_total'][true_rel] += 1
                if p == t:
                    metrics['per_class_correct'][true_rel] += 1
                
                # Check gender violations
                gender1, gender2 = batch['gender_features'][i]
                valid_rels = VALID_RELATIONSHIPS[(int(gender1.item()), int(gender2.item()))]
                
                if pred_rel not in valid_rels:
                    metrics['gender_violations'] += 1
                
                # Record errors
                if p != t or pred_rel not in valid_rels:
                    metrics['errors'].append({
                        'true_rel': true_rel,
                        'pred_rel': pred_rel,
                        'genders': (gender1.item(), gender2.item()),
                        'valid_rels': valid_rels,
                        'is_gender_violation': pred_rel not in valid_rels,
                        'anchor_path': metadata['anchor_path'][i],
                        'positive_path': metadata['positive_path'][i]
                    })
    
    # Calculate final metrics
    metrics['accuracy'] = metrics['correct'] / metrics['total']
    metrics['loss'] = metrics['loss'] / len(loader)
    metrics['gender_violation_rate'] = metrics['gender_violations'] / metrics['total']
    
    # Calculate per-class metrics
    metrics['per_class_accuracy'] = {
        rel: metrics['per_class_correct'][rel] / metrics['per_class_total'][rel]
        if metrics['per_class_total'][rel] > 0 else 0
        for rel in RELATIONSHIP_TYPES
    }
    
    # Calculate F1 score
    metrics['f1'] = f1_score(
        metrics['true_labels'],
        metrics['predictions'],
        average='weighted'
    )
    
    # Save detailed evaluation results if in test mode
    if mode == 'test':
        save_dir = Path('results/kin_relationship_v2')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / 'test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_samples': metrics['total'],
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1'],
                    'gender_violation_rate': metrics['gender_violation_rate'],
                    'per_class_accuracy': metrics['per_class_accuracy']
                },
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
                'errors': metrics['errors']
            }, f, indent=2)
    
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'analyze'])
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'use_wandb': args.use_wandb
    }
    
    # Data loading
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train'
    splits_to_load = ['test'] if args.mode == 'test' else ['train', 'val', 'test']
    
    print("\nLoading datasets...")
    data_splits = {}
    for split in splits_to_load:
        csv_path = f"{base_path}/splits_no_overlap_hand/{split}_triplets_enhanced.csv"
        data_splits[split] = pd.read_csv(csv_path)
        print(f"Loaded {split} split: {len(data_splits[split])} samples")
    
    # Load gender information
    metadata_path = os.path.join(base_path, 'fiw_metadata_filtered.csv')
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[~metadata_df['Aligned_Image_Path'].str.contains('unrelated')]
    
    for col in ['Aligned_Image_Path']:
        metadata_df[col] = metadata_df[col].str.replace(
            'data/',
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/',
            regex=False
        )
    
    gender_dict = {
        row['Aligned_Image_Path']: 1 if row['True_Gender'].lower() == 'm' else 0
        for _, row in metadata_df.iterrows()
    }
    
    # Create datasets and dataloaders
    datasets = {
        split: ImprovedRelationshipDataset(
            data_splits[split],
            gender_dict,
            is_training=(split=='train')
        )
        for split in splits_to_load
    }
    
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=config['batch_size'],
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        for split in splits_to_load
    }
    
    # Initialize model
    print("\nInitializing model...")
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    model = ImprovedRelationshipClassifier(onnx_path)
    
    if args.mode == 'train':
        print("\nStarting training...")
        best_metrics = train_model_improved(model, dataloaders['train'], dataloaders['val'], config)
        print(f"\nTraining completed. Best metrics: {best_metrics}")
    
    elif args.mode == 'test':
        print("\nLoading best model for testing...")
        checkpoint = torch.load('checkpoints/kin_relationship_v2/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        criterion = ImprovedGenderAwareLoss()
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model_improved(model, dataloaders['test'], criterion, device, mode='test')
        
        print("\nTest Results:")
        print(f"Overall Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        print(f"Gender Violation Rate: {test_metrics['gender_violation_rate']:.4f}")
        print("\nPer-Class Accuracy:")
        for rel, acc in test_metrics['per_class_accuracy'].items():
            print(f"{rel}: {acc:.4f} ({test_metrics['per_class_total'][rel]} samples)")
    
    elif args.mode == 'analyze':
        print("\nLoading best model for analysis...")
        checkpoint = torch.load('checkpoints/kin_relationship_v2/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        criterion = ImprovedGenderAwareLoss()
        
        print("\nAnalyzing model performance...")
        test_metrics = evaluate_model_improved(model, dataloaders['test'], criterion, device, mode='test')
        
        # Analyze error patterns
        errors = test_metrics['errors']
        error_patterns = {}
        for error in errors:
            key = (error['true_rel'], error['pred_rel'], error['is_gender_violation'])
            if key not in error_patterns:
                error_patterns[key] = []
            error_patterns[key].append(error)
        
        print("\nMost Common Error Patterns:")
        for (true_rel, pred_rel, is_violation), examples in sorted(
            error_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"\n{true_rel} -> {pred_rel} (Gender violation: {is_violation})")
            print(f"Total occurrences: {len(examples)}")
            print("Example cases:")
            for example in examples[:3]:
                print(f"- Genders: {example['genders']}")
                print(f"  Valid relationships: {example['valid_rels']}")
                print(f"  Anchor: {example['anchor_path']}")
                print(f"  Positive: {example['positive_path']}")
        
        # Analyze challenging pairs
        print("\nAnalyzing Most Challenging Relationship Pairs...")
        confusion_matrix = test_metrics['confusion_matrix']
        total_per_class = confusion_matrix.sum(axis=1)
        confusion_rate = confusion_matrix / total_per_class[:, np.newaxis]
        
        for i in range(len(RELATIONSHIP_TYPES)):
            for j in range(len(RELATIONSHIP_TYPES)):
                if i != j and confusion_rate[i, j] > 0.1:  # Show pairs with >10% confusion
                    print(f"\n{IDX_TO_REL[i]} confused with {IDX_TO_REL[j]}: "
                          f"{confusion_rate[i, j]*100:.1f}% of cases")


            
        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()
    
# Analyzing Most Challenging Relationship Pairs...

# ss confused with md: 38.4% of cases
# bb confused with fs: 34.9% of cases
# sibs confused with ms: 14.1% of cases
# sibs confused with fd: 27.9% of cases