import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
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
from insightface.app import FaceAnalysis
import warnings
import pickle
warnings.filterwarnings('ignore')

# set device cuda 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        # Global backbone - frozen InsightFace
        onnx_model = onnx.load(onnx_path)
        self.backbone = convert(onnx_model)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.local_projector = nn.Linear(256, 512)

        # Local feature extractors with shared weights
        self.local_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Cross-modal attention
        self.cross_modal_attention = nn.ModuleDict({
            'global_to_local': nn.MultiheadAttention(512, 8, dropout=0.2),
            'local_to_global': nn.MultiheadAttention(512, 8, dropout=0.2)
        })
        
        # Feature interaction module
        # Feature interaction module
        self.interaction = nn.Sequential(
            nn.Linear(512 * 4, 1024),  # Changed from 512 * 6 to 512 * 4
            nn.LayerNorm(1024),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.PReLU(),
            nn.Dropout(0.3)
        )
        
        # Siamese projector
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Linear(256, 256)
        )
        
        # Auxiliary age difference predictor
        self.age_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )
        
        # Auxiliary gender predictor
        self.gender_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 2)
        )
    
    def extract_region(self, x, center, size):
        batch_size = x.size(0)
        patches = []
        
        for b in range(batch_size):
            x_center, y_center = center[b]
            half_size = size // 2
            
            x_start = max(0, int(x_center - half_size))
            y_start = max(0, int(y_center - half_size))
            x_end = min(x.size(3), x_start + size)
            y_end = min(x.size(2), y_start + size)
            
            patch = x[b:b+1, :, y_start:y_end, x_start:x_end]
            patch = F.interpolate(patch, size=(size, size), mode='bilinear', align_corners=True)
            patches.append(patch)
        
        return torch.cat(patches, dim=0)

    def extract_local_features(self, x, kps):
        batch_size = x.size(0)
        local_features = []
        
        # Extract and encode local regions
        regions = {
            'eyes': (kps[:, 0:2].mean(dim=1), 40),
            'nose': (kps[:, 2], 32),
            'mouth': (kps[:, 3:5].mean(dim=1), 32)
        }
        
        for center, size in regions.values():
            region = self.extract_region(x, center, size)
            local_features.append(self.local_encoder(region))
            
        return torch.stack(local_features, dim=1)  # [B, 3, 256]
    
    def forward_one(self, x, kps):
        # Global features from backbone
        global_features = self.backbone(x)  # [B, 512]
        
        # Extract and encode local features
        # Extract and encode local features
        local_features = self.extract_local_features(x, kps)  # [B, 3, 256]
        local_features = local_features.mean(dim=1)  # [B, 256]
        local_features = self.local_projector(local_features)  # Project to 512 dims
        # Cross-modal attention
        global_context, _ = self.cross_modal_attention['global_to_local'](
            global_features.unsqueeze(0),
            local_features.unsqueeze(0),
            local_features.unsqueeze(0)
        )
        local_context, _ = self.cross_modal_attention['local_to_global'](
            local_features.unsqueeze(0),
            global_features.unsqueeze(0),
            global_features.unsqueeze(0)
        )
        
        # Concatenate all features
        # Concatenate all features
        combined = torch.cat([
            global_features,  # [B, 512]
            local_features,   # [B, 512]
            global_context.squeeze(0),  # [B, 512]
            local_context.squeeze(0)    # [B, 512]
        ], dim=1)  # Total: [B, 2048]
        
        # Feature interaction and projection
        interacted = self.interaction(combined)
        projected = self.projector(interacted)
        
        # Auxiliary predictions
        age_pred = self.age_predictor(interacted)
        gender_pred = self.gender_predictor(interacted)
        
        return {
            'embedding': F.normalize(projected, p=2, dim=1),
            'age_pred': age_pred,
            'gender_pred': gender_pred,
            'features': interacted
        }
    
    def forward(self, x1, x2, kps1, kps2):
        out1 = self.forward_one(x1, kps1)
        out2 = self.forward_one(x2, kps2)
        return out1, out2
        
class MultiTaskLoss(nn.Module):
    def __init__(self, margin=0.3, temperature=0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, out1, out2, labels, age_diff=None, gender1=None, gender2=None):
        emb1, emb2 = out1['embedding'], out2['embedding']
        
        # Contrastive loss
        similarity = F.cosine_similarity(emb1, emb2)
        contrastive_loss = labels * torch.pow(1 - similarity, 2) + \
                          (1 - labels) * torch.pow(torch.clamp(similarity - self.margin, min=0.0), 2)
        
        # InfoNCE loss
        batch_size = emb1.size(0)
        features = torch.cat([emb1, emb2], dim=0)
        sim_matrix = torch.matmul(features, features.T)
        
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=sim_matrix.device)
        positives = sim_matrix[mask].view(batch_size * 2, 1)
        negatives = sim_matrix[~mask].view(batch_size * 2, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        infonce_labels = torch.zeros(batch_size * 2, device=logits.device, dtype=torch.long)
        infonce_loss = self.ce_loss(logits / self.temperature, infonce_labels)
        
        # Auxiliary losses
        loss = contrastive_loss.mean() + 0.5 * infonce_loss
        
        if age_diff is not None:
            age_loss = self.mse_loss(out1['age_pred'], age_diff) + \
                      self.mse_loss(out2['age_pred'], age_diff)
            loss = loss + 0.1 * age_loss
            
        if gender1 is not None and gender2 is not None:
            gender_loss = self.ce_loss(out1['gender_pred'], gender1) + \
                         self.ce_loss(out2['gender_pred'], gender2)
            loss = loss + 0.1 * gender_loss
        
        return loss
class KinshipDataset(Dataset):
    def __init__(self, triplets_df, split_name, transform=None, is_training=True):
        self.triplets_df = triplets_df
        self.is_training = is_training
        
        # Load preprocessed keypoints
        keypoints_path = f'/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2/keypoints/{split_name}_keypoints.pkl'
        with open(keypoints_path, 'rb') as f:
            self.keypoints_dict = pickle.load(f)
        
        # Update paths
        for col in ['Anchor', 'Positive', 'Negative']:
            self.triplets_df[col] = self.triplets_df[col].str.replace(
                '../data',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
                regex=False
            )
        
        # Define transformations
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    
    def load_image(self, image_path):
        """Load image and get cached keypoints"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                kps = torch.tensor(self.keypoints_dict[image_path])
                return img, kps
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.1)
    
    def __len__(self):
        return len(self.triplets_df) * 2
    
    def __getitem__(self, idx):
        row_idx = idx // 2
        is_positive = idx % 2 == 0
        
        row = self.triplets_df.iloc[row_idx]
        
        try:
            # Load images and cached keypoints
            anchor_img, anchor_kps = self.load_image(row['Anchor'])
            pair_img, pair_kps = self.load_image(row['Positive' if is_positive else 'Negative'])
            
            # Apply transformations
            anchor_tensor = self.transform(anchor_img)
            pair_tensor = self.transform(pair_img)
            
            # Scale keypoints to match transformed image size
            h_scale = 112 / anchor_img.shape[0]
            w_scale = 112 / anchor_img.shape[1]
            
            anchor_kps = anchor_kps * torch.tensor([[w_scale, h_scale]])
            pair_kps = pair_kps * torch.tensor([[w_scale, h_scale]])
            
            return {
                'anchor': anchor_tensor,
                'pair': pair_tensor,
                'anchor_kps': anchor_kps,
                'pair_kps': pair_kps,
                'is_kin': torch.LongTensor([1 if is_positive else 0])
            }
        except Exception as e:
            print(f"Error loading images for row {row_idx}: {str(e)}")
            raise

def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Finding optimal threshold'):
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            anchor_kps = batch['anchor_kps'].to(device)
            pair_kps = batch['pair_kps'].to(device)
            labels = batch['is_kin'].to(device)
            
            # In find_optimal_threshold and evaluate:
            out1, out2 = model(anchor, pair, anchor_kps, pair_kps)
            similarities = F.cosine_similarity(out1['embedding'], out2['embedding'])
            
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    similarities = np.array(all_similarities)
    labels = np.array(all_labels)
    
    # Calculate ROC curve and find best threshold
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    auc_score = auc(fpr, tpr)
    
    # Calculate accuracy at optimal threshold
    predictions = (similarities >= optimal_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    
    return optimal_threshold, auc_score, accuracy


def train_model(model, train_loader, val_loader, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler()  # Remove 'cuda' argument
    
    criterion = MultiTaskLoss(margin=config['margin'], temperature=config['temperature'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    num_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config['warmup_epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['lr'],
        total_steps=num_steps,
        pct_start=warmup_steps/num_steps
    )
    
    best_val = {'accuracy': 0, 'epoch': 0, 'state': None, 'threshold': None, 'auc': None}
    patience_counter = 0
    current_threshold = 0.5
    
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            anchor_kps = batch['anchor_kps'].to(device)
            pair_kps = batch['pair_kps'].to(device)
            is_kin = batch['is_kin'].to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass
                out1, out2 = model(anchor, pair, anchor_kps, pair_kps)
                loss = criterion(out1, out2, is_kin)
                
                # Calculate accuracy
                with torch.no_grad():
                    similarities = F.cosine_similarity(out1['embedding'], out2['embedding'])
                    predictions = (similarities > current_threshold).long()
                    correct = (predictions == is_kin.view(-1)).sum().item()
                    running_correct += correct
                    running_total += is_kin.size(0)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update running loss and accuracy
            running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
            current_acc = (running_correct / running_total) * 100 if running_total > 0 else 0.0
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss:.4f}',
                'acc': f'{current_acc:.2f}%',
                'threshold': f'{current_threshold:.4f}'
            })
        
        # Calculate final training metrics
        train_acc = (running_correct / running_total) * 100 if running_total > 0 else 0.0
        
        # Validation phase
        model.eval()
        optimal_threshold, val_auc, val_acc = find_optimal_threshold(model, val_loader, device)
        val_acc = val_acc * 100  # Convert to percentage
        
        # Update threshold for next epoch
        current_threshold = optimal_threshold
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training Loss: {running_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Validation AUC: {val_auc:.4f}')
        print(f'Optimal Threshold: {optimal_threshold:.4f}')
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': running_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'optimal_threshold': optimal_threshold,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_acc > best_val['accuracy']:
            best_val = {
                'accuracy': val_acc,
                'epoch': epoch,
                'state': model.state_dict().copy(),
                'threshold': optimal_threshold,
                'auc': val_auc
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'threshold': optimal_threshold,
                'auc': val_auc,
                'config': config
            }, 'checkpoints/kin_binary_v3n/best_model.pth')
            print(f'New best model saved! Validation Accuracy: {val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'Early stopping triggered after {patience_counter} epochs without improvement')
                wandb.log({'early_stopping_epoch': epoch + 1})
                break
        
        print('-' * 60)
    
    return best_val


def evaluate(model, dataloader, device, threshold=None):
    model.eval()
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            anchor_kps = batch['anchor_kps'].to(device)
            pair_kps = batch['pair_kps'].to(device)
            is_kin = batch['is_kin'].to(device)
            
            out1, out2 = model(anchor, pair, anchor_kps, pair_kps)
            similarities = F.cosine_similarity(out1['embedding'], out2['embedding'])
            
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(is_kin.cpu().numpy())
    
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    if threshold is None:
        # Find optimal threshold if not provided
        fpr, tpr, thresholds = roc_curve(all_labels, all_similarities)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    predictions = (all_similarities >= threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    auc_score = roc_auc_score(all_labels, all_similarities)
    
    return accuracy, auc_score, threshold


def main():
    os.makedirs('checkpoints/kin_binary_v3n', exist_ok=True)
    # Configuration
    config = {
        'lr': 2e-6,
        'batch_size': 128,
        'epochs': 20,
        'warmup_epochs': 2,
        'patience': 5,
        'margin': 0.3,
        'temperature': 0.07,
        'weight_decay': 0.05,
        'label_smoothing': 0.1
    }
    
    # Initialize wandb
    wandb.init(
        project="kinship-verification",
        config=config
    )
    
    # Model initialization
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    model = KinshipVerificationModel(onnx_path)
    
    # Data loading
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2'
    
    train_df = pd.read_csv(os.path.join(base_path, 'train_triplets_enhanced.csv'))
    val_df = pd.read_csv(os.path.join(base_path, 'val_triplets_enhanced.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test_triplets_enhanced.csv'))
    
    print(f"Dataset sizes:")
    print(f"Train: {len(train_df)} triplets")
    print(f"Val: {len(val_df)} triplets")
    print(f"Test: {len(test_df)} triplets")
    
    # Create datasets
    train_dataset = KinshipDataset(train_df, split_name='train', is_training=True)
    val_dataset = KinshipDataset(val_df, split_name='val', is_training=False)
    test_dataset = KinshipDataset(test_df, split_name='test', is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train model
    best_val_results = train_model(model, train_loader, val_loader, test_loader, config)
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_val_results['state'])
    test_acc, test_auc, test_threshold = evaluate(model, test_loader, 'cuda', threshold=best_val_results['threshold'])

    # Print final results
    print("\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_results['accuracy']:.4f}")
    print(f"Best Validation AUC: {best_val_results['auc']:.4f}")
    print(f"Best Validation Epoch: {best_val_results['epoch']}")

    print('-' * 60)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Optimal Threshold: {test_threshold:.4f}")
    
    # Log final results to wandb
    wandb.log({
        'final_val_acc': best_val_results['accuracy'],
        'final_test_acc': test_acc,
        'best_epoch': best_val_results['epoch']
    })
    
    # Save final results
    results = {
        'best_val_accuracy': float(best_val_results['accuracy']),
        'best_epoch': int(best_val_results['epoch']),
        'test_accuracy': float(test_acc),
        'config': config,
        'timestamp': time.strftime('%Y%m%d-%H%M%S')
    }
    
    with open('final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    wandb.finish()

if __name__ == "__main__":
    main()