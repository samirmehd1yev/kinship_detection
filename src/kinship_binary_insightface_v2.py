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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import logging
from pathlib import Path
import threading
from queue import Queue
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
import argparse

def setup_logging(save_dir):
    log_file = save_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
# Set up logging
save_dir = Path('checkpoints/kin_binary_v2')
save_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(save_dir)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Reshape and transpose for attention calculation
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C
        proj_key = self.key(x).view(batch_size, -1, H * W)  # B x C x (H*W)
        energy = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, H * W)  # B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class KinshipVerificationModel(nn.Module):
    def __init__(self, onnx_path, embedding_dim=512, dropout_rate=0.2):
        super().__init__()
        self.feature_extractor = KinshipFeatureExtractor(
            onnx_path=onnx_path,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x1, x2):
        emb1 = self.feature_extractor.forward_features(x1)
        emb2 = self.feature_extractor.forward_features(x2)
        return emb1, emb2

class CombinedLoss(nn.Module):
    def __init__(self, margin=0.5, center_weight=0.1, triplet_weight=0.3):
        super().__init__()
        self.contrastive = ContrastiveLoss(margin)
        self.center = CenterLoss(center_weight)
        self.triplet = TripletLoss(margin)
        self.triplet_weight = triplet_weight
        
    def forward(self, emb1, emb2, labels, anchor_emb=None, neg_emb=None):
        losses = {
            'contrastive': self.contrastive(emb1, emb2, labels),
            'center': self.center(emb1, emb2, labels)
        }
        
        if anchor_emb is not None and neg_emb is not None:
            losses['triplet'] = self.triplet_weight * self.triplet(anchor_emb, emb1, neg_emb)
        
        total_loss = sum(losses.values())
        return total_loss, losses

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, emb1, emb2, labels):
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        similarity = F.cosine_similarity(emb1, emb2)
        labels = labels.float()
        
        loss = labels * torch.pow(1 - similarity, 2) + \
               (1 - labels) * torch.pow(torch.clamp(similarity - self.margin, min=0.0), 2)
               
        return loss.mean()

class CenterLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        self.centers = None
        
    def forward(self, emb1, emb2, labels):
        if self.centers is None:
            self.centers = nn.Parameter(torch.randn(2, emb1.size(1), device=emb1.device))
            
        centers = self.centers[labels]
        center_loss = (torch.sum((emb1 - centers)**2) + 
                      torch.sum((emb2 - centers)**2)) / emb1.size(0)
                      
        return self.weight * center_loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_pos = (anchor - positive).pow(2).sum(1)
        distance_neg = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_pos - distance_neg + self.margin)
        return losses.mean()

class KinshipDataset(Dataset):
    def __init__(self, triplets_df, transform=None, is_training=True):
        self.triplets_df = triplets_df
        self.is_training = is_training
        
        # Update paths
        for col in ['Anchor', 'Positive', 'Negative']:
            self.triplets_df[col] = self.triplets_df[col].str.replace(
                '../data',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
                regex=False
            )
        
        if transform is None:
            if is_training:
                self.transform = A.Compose([
                    A.Transpose(p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=45,
                        p=0.2
                    ),
                    A.OneOf([
                        A.MotionBlur(p=.2),
                        A.MedianBlur(blur_limit=3, p=.1),
                        A.Blur(blur_limit=3, p=.1),
                    ], p=0.2),
                    A.OneOf([
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=.1),
                    ], p=0.2),
                    A.OneOf([
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),
                    ], p=0.3),
                    A.HueSaturationValue(p=0.3),
                    A.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                    ),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                    ),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform
            
        # Preload images in memory if possible
        self.image_cache = {}
        self.preload_images()
        
    def preload_images(self):
        def load_worker(paths, queue):
            for path in paths:
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        queue.put((path, img))
                except Exception as e:
                    logging.error(f"Error loading image {path}: {str(e)}")
                    
        unique_paths = pd.unique(self.triplets_df[['Anchor', 'Positive', 'Negative']].values.ravel())
        num_threads = 4
        chunk_size = len(unique_paths) // num_threads
        
        threads = []
        queue = Queue()
        
        for i in range(num_threads):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_threads - 1 else len(unique_paths)
            thread = threading.Thread(
                target=load_worker,
                args=(unique_paths[start_idx:end_idx], queue)
            )
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        while not queue.empty():
            path, img = queue.get()
            self.image_cache[path] = img
    
    def __len__(self):
        return len(self.triplets_df) * 2
    
    def load_image(self, image_path):
        if image_path in self.image_cache:
            img = self.image_cache[image_path]
        else:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        raise ValueError(f"Could not load image: {image_path}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.1)
                    
        transformed = self.transform(image=img)
        return transformed['image']
    
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
            
            # Validation checks
            assert torch.isfinite(anchor_img).all(), f"Found inf/nan in anchor image at idx {idx}"
            assert torch.isfinite(pair_img).all(), f"Found inf/nan in pair image at idx {idx}"
            
            return {
                'anchor': anchor_img,
                'pair': pair_img,
                'is_kin': torch.LongTensor([label])
            }
        except Exception as e:
            logging.error(f"Error loading images for row {row_idx}: {str(e)}")
            raise

class MultiScaleFeatures(nn.Module):
    def __init__(self, scales=[0.75, 1.25], base_size=112):
        super().__init__()
        self.scales = scales
        self.base_size = base_size
        
    def forward(self, x):
        features = []
        # Process original size
        features.append(x)
        
        # Process scaled versions
        for scale in self.scales:
            size = int(self.base_size * scale)
            scaled = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
            # Resize back to original size
            scaled = F.interpolate(scaled, size=(self.base_size, self.base_size), 
                                 mode='bilinear', align_corners=False)
            features.append(scaled)
            
        # Concatenate along channel dimension
        return torch.cat(features, dim=1)

class KinshipFeatureExtractor(nn.Module):
    def __init__(self, onnx_path, embedding_dim=512, dropout_rate=0.2):
        super().__init__()
        self.backbone = convert(onnx.load(onnx_path))
        self.backbone.requires_grad_(True)
        
        # Enable gradient checkpointing
        self.use_gradient_checkpointing = True
        
        # Feature refinement using fully connected layers
        self.feature_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism for feature weighting
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )
        
        # Final embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward_features(self, x):
        # Get backbone features
        if self.use_gradient_checkpointing and self.training:
            features = checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        else:
            features = self.backbone(x)
        
        # Refine features
        refined = self.feature_layers(features)
        
        # Apply attention
        attention_weights = self.attention(refined)
        attended_features = refined * attention_weights
        
        # Get final embedding
        embedding = self.embedding(attended_features)
        
        return F.normalize(embedding, p=2, dim=1)


class OnlineHardNegativeMining:
    def __init__(self, model, train_loader, device, k=1000):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.k = k
        self.negative_pool = []
        
    def update_negative_pool(self):
        self.model.eval()
        embeddings = []
        paths = []
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Updating negative pool"):
                anchor = batch['anchor'].to(self.device)
                emb = self.model(anchor, anchor)[0]  # Using the same image for both inputs
                embeddings.append(emb.cpu())
                
                # Assuming the paths are available in the dataset
                if 'anchor_path' in batch:
                    paths.extend(batch['anchor_path'])
                else:
                    # If paths are not available, use indices as identifiers
                    paths.extend([str(i) for i in range(len(emb))])
        
        embeddings = torch.cat(embeddings)
        self.negative_pool = list(zip(paths, embeddings))
        random.shuffle(self.negative_pool)
        self.negative_pool = self.negative_pool[:self.k]

def evaluate_model(model, data_loader, device, threshold=None):
    model.eval()
    predictions = []
    labels = []
    similarities = []
    
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
            
            similarities.extend(similarity.cpu().numpy())
            labels.extend(is_kin.numpy())
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Calculate AUC
    auc = roc_auc_score(labels, similarities)
    
    # Find best threshold if not provided
    if threshold is None:
        thresholds = np.arange(-1.0, 1.0, 0.01)
        best_acc = 0
        best_threshold = 0
        
        for t in thresholds:
            preds = (similarities >= t).astype(int)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
        threshold = best_threshold
    
    # Calculate metrics with chosen threshold
    predictions = (similarities >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'auc': auc,
        'f1': f1_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'threshold': threshold
    }
    
    return metrics

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def train_model(model, train_loader, val_loader, test_loader, config, save_dir):
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize mining and losses
    hard_mining = OnlineHardNegativeMining(model, train_loader, device)
    criterion = CombinedLoss(
        margin=config['optimization']['margin'],
        center_weight=config['optimization']['center_weight'],
        triplet_weight=config['optimization']['triplet_weight']
    ).to(device)
    
    # Initialize optimizer with different LRs for backbone and new layers
    backbone_params = list(model.feature_extractor.backbone.parameters())
    new_params = (
        list(model.feature_extractor.feature_layers.parameters()) +
        list(model.feature_extractor.attention.parameters()) +
        list(model.feature_extractor.embedding.parameters())
    )
                
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['optimization']['lr_backbone']},
        {'params': new_params, 'lr': config['optimization']['lr_new']}
    ], weight_decay=config['optimization']['weight_decay'])
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Setup learning rate schedule
    num_training_steps = len(train_loader) * config['training']['epochs']
    num_warmup_steps = len(train_loader) * config['training']['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # Initialize best metrics tracking
    best_val_metrics = {
        'accuracy': 0,
        'auc': 0,
        'epoch': 0,
        'model_state': None,
        'threshold': 0
    }
    
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0
        loss_components = defaultdict(float)
        batch_metrics = defaultdict(list)
        num_batches = 0
        
        # Update negative mining pool periodically
        if epoch % config['training']['mining_frequency'] == 0:
            logger.info("Updating negative mining pool...")
            hard_mining.update_negative_pool()
            
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            labels = batch['is_kin'].to(device)
            
            # Mixed precision training
            with autocast('cuda'):
                emb1, emb2 = model(anchor, pair)
                loss, loss_dict = criterion(emb1, emb2, labels)
                
            # Update loss components
            for k, v in loss_dict.items():
                loss_components[k] += v.item()
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                similarity = F.cosine_similarity(emb1, emb2)
                preds = (similarity > best_val_metrics['threshold']).float()
                acc = (preds == labels.float()).float().mean()
                
                batch_metrics['accuracy'].append(acc.item())
                batch_metrics['loss'].append(loss.item())
            
            num_batches += 1
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if batch_idx % config['training']['log_frequency'] == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_acc': acc.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    **{f'loss_{k}': v/num_batches for k, v in loss_components.items()}
                })
        
        # Evaluation phase
        val_metrics = evaluate_model(model, val_loader, device, threshold=best_val_metrics['threshold'])
        
        # Logging
        epoch_loss = running_loss / num_batches
        epoch_acc = np.mean(batch_metrics['accuracy'])
        
        logger.info(f'\nEpoch {epoch+1} Summary:')
        logger.info(f'Training - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        logger.info(f'Validation - Acc: {val_metrics["accuracy"]:.4f}, AUC: {val_metrics["auc"]:.4f}')
        
        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_acc': val_metrics['accuracy'],
            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall']
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_metrics['accuracy']:
            logger.info(f"New best model with validation accuracy: {val_metrics['accuracy']:.4f}")
            best_val_metrics.update({
                'accuracy': val_metrics['accuracy'],
                'auc': val_metrics['auc'],
                'epoch': epoch,
                'model_state': model.state_dict(),
                'threshold': val_metrics['threshold']
            })
            
            save_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, save_path)
    
    return best_val_metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.json')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    
    # Set random seed
    set_seed(config['seed'])
    
    # Initialize wandb
    wandb.init(
        project=config['wandb_project'],
        name=config['run_name'],
        config=config
    )
    
    # Load model
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    model = KinshipVerificationModel(
        onnx_path,
        embedding_dim=config['model']['embedding_dim'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Load data
    train_data = pd.read_csv(config['train_path'])
    val_data = pd.read_csv(config['val_path'])
    test_data = pd.read_csv(config['test_path'])
    
    # Create datasets
    train_dataset = KinshipDataset(train_data, is_training=True)
    val_dataset = KinshipDataset(val_data, is_training=False)
    test_dataset = KinshipDataset(test_data, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    if args.test_only:
        # Load best model and evaluate
        checkpoint = torch.load(save_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Running final evaluation...")
        test_metrics = evaluate_model(model, test_loader, torch.device('cuda'))
        
        logger.info("\nTest Set Results:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        results_file = save_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_metrics, f, indent=4)
            
        return
    
    # Train model
    best_val_metrics = train_model(
        model, train_loader, val_loader, test_loader, 
        config, save_dir
    )
    
    # Final evaluation
    logger.info("\nTraining completed! Loading best model for final evaluation...")
    model.load_state_dict(best_val_metrics['model_state'])
    
    test_metrics = evaluate_model(
        model, test_loader, torch.device('cuda'),
        threshold=best_val_metrics['threshold']
    )
    
    logger.info("\nFinal Test Set Results:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save final results
    final_results = {
        'best_val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'config': config
    }
    
    results_file = save_dir / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    wandb.finish()

if __name__ == "__main__":
    main()