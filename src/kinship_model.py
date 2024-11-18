import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
from tqdm import tqdm
import os
import json
import time
from PIL import Image
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from collections import defaultdict
import torch.nn.functional as F # Import torchvision models
from torchvision import models

import torch.nn as nn
from torchvision import transforms
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve, auc

class Config:
    def __init__(self):
        # Base path
        self.base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model'
        
        # Create experiment name with timestamp
        self.experiment_name = f'kinship_verification_model_18nov'
        
        # Data settings
        self.train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
        self.val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
        self.test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'
        
        # Add memory management settings
        self.empty_cache_freq = 50
        self.mixed_precision = True
        self.max_grad_norm = 1.0
        
        # Add monitoring settings
        self.log_embeddings_freq = 5
        self.log_batch_freq = 100
        self.log_dist_freq = 500
        
        # Add training optimizations
        self.label_smoothing = 0.1
        self.warmup_steps = 1000
        self.use_amp = True  # Automatic Mixed Precision
        
        # Add data augmentation settings
        self.use_random_crop = False
        self.use_color_jitter = False
        self.use_random_flip = True
        self.jitter_brightness = 0.2
        self.jitter_contrast = 0.2
        self.jitter_saturation = 0.2
        self.jitter_hue = 0.1
        
        # Training settings
        self.num_workers = 4 
        self.use_cache = False
        self.batch_size = 16  # Reduced from 32
        self.gradient_accumulation_steps = 2  # Added gradient accumulation
        self.num_epochs = 15
        self.learning_rate = 2e-4
        self.weight_decay = 1e-4
        
        # Memory management
        self.pin_memory = False  # Disabled pin_memory to reduce memory usage
        
        # Model settings
        self.embedding_dim = 256
        self.margin = 0.5
        self.focal_gamma = 2.0
        
        # Early stopping settings
        self.patience = 5
        self.min_delta = 0.001
        
        # Learning rate scheduler settings
        self.min_lr = 1e-6
        self.warmup_epochs = 5
        self.T_max = 20
        
        # Directories setup
        self._setup_directories()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Save config
        self.save_config()
    
    def _setup_directories(self):
        """Setup directory structure for the experiment"""
        self.experiment_dir = os.path.join(self.base_path, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.results_dir = os.path.join(self.experiment_dir, 'results')
        
        # Create all directories
        for directory in [self.experiment_dir, self.checkpoint_dir, 
                         self.log_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up model paths
        self.best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        self.last_model_path = os.path.join(self.checkpoint_dir, 'last_model.pth')
        self.config_path = os.path.join(self.experiment_dir, 'config.json')
    
    def save_config(self):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in vars(self).items() 
                      if not k.startswith('_') and not isinstance(v, torch.device)}
        config_dict['device'] = str(self.device)
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

from torch.multiprocessing import Value
import ctypes

class StepCounter:
    def __init__(self):
        self.value = Value(ctypes.c_int64, 0)
    
    def increment(self):
        with self.value.get_lock():
            self.value.value += 1
            return self.value.value
    
    def get(self):
        return self.value.value

class ImageProcessor:
    def __init__(self, train_mode=True, config=None):
        self.train_mode = train_mode
        self.config = config
        self.processing_stats = defaultdict(int)
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if train_mode and config:
            if config.use_random_flip:
                self.transform.transforms.insert(1, transforms.RandomHorizontalFlip())
            if config.use_random_crop:
                self.transform.transforms.insert(1, transforms.RandomResizedCrop(112, scale=(0.8, 1.0)))
            if config.use_color_jitter:
                self.transform.transforms.insert(1, transforms.ColorJitter(
                    brightness=config.jitter_brightness,
                    contrast=config.jitter_contrast,
                    saturation=config.jitter_saturation,
                    hue=config.jitter_hue
                ))
    
    def process_face(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            transformed_img = self.transform(img)
            
            self.processing_stats['total_processed'] += 1
            self.processing_stats['successful_processed'] += 1
            
            return transformed_img
            
        except Exception as e:
            self.processing_stats['failed_processed'] += 1
            print(f"Error processing image {image_path}: {str(e)}")
            return None


def save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, train_metrics, best_val_f1, config):
    """Save model checkpoint with all necessary information"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_f1': best_val_f1,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': vars(config)
    }
    torch.save(checkpoint, config.best_model_path)
    
    # Create a backup with metrics in filename
    best_model_backup = os.path.join(
        config.checkpoint_dir, 
        f'best_model_f1_{best_val_f1:.4f}_epoch_{epoch}.pth'
    )
    shutil.copy2(config.best_model_path, best_model_backup)

def print_metrics(train_metrics, val_metrics, current_lr):
    """Print training and validation metrics in a formatted way"""
    print("\nTraining Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCurrent learning rate: {current_lr:.6f}")

def cleanup_and_log_final_metrics(model, val_loader, criterion, config, best_model_state, best_val_f1, global_step):
    """Log final metrics and cleanup"""
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        final_metrics = validate(model, val_loader, criterion, config, -1, global_step)
        
        wandb.run.summary.update({
            "final_val_loss": final_metrics['val_loss'],
            "final_val_accuracy": final_metrics['val_accuracy'],
            "final_val_f1": final_metrics.get('val_f1', 0),
            "best_val_f1": best_val_f1,
            "final_step": global_step
        })
        
        print("\nTraining completed!")
        print(f"Best validation F1: {best_val_f1:.4f}")
        print("Final validation metrics:", final_metrics)     
def log_wandb_metrics(epoch_metrics, epoch, current_lr, prefix='', step=None):
    """Log metrics to wandb with proper step tracking"""
    metrics = {
        f'{prefix}/loss': epoch_metrics[f'{prefix}_loss'],
        f'{prefix}/accuracy': epoch_metrics[f'{prefix}_accuracy'],
        'learning_rate': current_lr,
        'epoch': epoch
    }
    
    if 'val' in prefix:
        metrics.update({
            f'{prefix}/precision': epoch_metrics[f'{prefix}_precision'],
            f'{prefix}/recall': epoch_metrics[f'{prefix}_recall'],
            f'{prefix}/f1': epoch_metrics[f'{prefix}_f1']
        })
    
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)
def log_wandb_embeddings(model, val_loader, config, global_step):
    """Log embeddings visualization to wandb with step tracking"""
    if global_step % (5 * len(val_loader)) == 0:
        model.eval()
        embeddings = []
        labels = []
        images = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 100:
                    break
                    
                anchor = batch['anchor'].to(config.device)
                positive = batch['positive'].to(config.device)
                negative = batch['negative'].to(config.device)
                
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                
                embeddings.extend(anchor_emb.cpu().numpy())
                embeddings.extend(positive_emb.cpu().numpy())
                embeddings.extend(negative_emb.cpu().numpy())
                
                labels.extend([0] * anchor_emb.size(0))
                labels.extend([1] * positive_emb.size(0))
                labels.extend([2] * negative_emb.size(0))
                
                images.extend([wandb.Image(img) for img in anchor[:5].cpu()])
        
        wandb.log({
            "embeddings": wandb.Table(
                columns=["embedding", "label", "step"],
                data=[[emb, label, global_step] for emb, label in zip(embeddings, labels)]
            ),
            "sample_images": images[:25]
        }, step=global_step)

def log_wandb_distance_distributions(pos_dists, neg_dists, step):
    """Log distance distributions to wandb with step tracking"""
    distances = np.concatenate([pos_dists.cpu().numpy(), neg_dists.cpu().numpy()])
    table = wandb.Table(data=[[d] for d in distances], columns=["Distances"])
    wandb.log({
        "distance_distributions": wandb.plot.histogram(table, "Distances", title="Distance Distributions")
    }, step=step)

def analyze_architecture_mismatch(model_state_dict, pretrained_state_dict):
    """
    Analyzes and prints detailed information about architecture mismatches
    """
    print("\nAnalyzing architecture differences:")
    
    # Clean pretrained state dict keys
    cleaned_pretrained = {}
    for k, v in pretrained_state_dict.items():
        clean_k = k.replace('backbone.', '').replace('module.', '')
        cleaned_pretrained[clean_k] = v
    
    # Compare keys and shapes
    model_keys = set(model_state_dict.keys())
    pretrained_keys = set(cleaned_pretrained.keys())
    
    # Find missing and extra keys
    missing_keys = model_keys - pretrained_keys
    extra_keys = pretrained_keys - model_keys
    
    # Analyze shape mismatches
    shape_mismatches = []
    for k in model_keys.intersection(pretrained_keys):
        if model_state_dict[k].shape != cleaned_pretrained[k].shape:
            shape_mismatches.append((k, model_state_dict[k].shape, cleaned_pretrained[k].shape))
    
    # Print results
    print("\nMissing keys in pretrained weights:")
    for k in sorted(missing_keys):
        print(f"  {k}: {model_state_dict[k].shape}")
    
    print("\nExtra keys in pretrained weights:")
    for k in sorted(extra_keys):
        print(f"  {k}: {cleaned_pretrained[k].shape}")
    
    print("\nShape mismatches:")
    for k, model_shape, pretrained_shape in shape_mismatches:
        print(f"  {k}: Model {model_shape} vs Pretrained {pretrained_shape}")
    
    return len(missing_keys), len(extra_keys), len(shape_mismatches)



class IRBlock(nn.Module):
    """Basic IR block for ResNet"""
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        # BatchNorm -> Conv -> BatchNorm -> PReLU -> Conv -> BatchNorm
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.prelu = nn.PReLU(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        # Residual connection with optional downsampling
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        # Main path
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class IResNet(nn.Module):
    """Improved ResNet architecture for face recognition"""
    def __init__(self, input_size=112):
        super().__init__()
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        
        # Layer configurations for ResNet-50
        blocks = [3, 4, 14, 3]
        
        # Main layers with stride=2 for downsampling
        self.layer1 = self._make_layer(IRBlock, 64, 64, blocks[0], stride=2)
        self.layer2 = self._make_layer(IRBlock, 64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(IRBlock, 128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(IRBlock, 256, 512, blocks[3], stride=2)
        
        # Final layers
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)
        
        # Calculate feature size: input_size/(2^4) for 4 stride-2 layers
        self.feature_size = (input_size // 16) ** 2 * 512
        self.fc = nn.Linear(self.feature_size, 512)
        self.features = nn.BatchNorm1d(512)

        self._initialize_weights()
        print(f"Model initialized with feature size: {self.feature_size}")

    def _make_layer(self, block, in_channel, out_channel, blocks, stride=1):
        """Create a layer with given number of blocks"""
        layers = []
        # First block with stride and potential channel change
        layers.append(block(in_channel, out_channel, stride))
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass with shape logging for debugging"""
        # Initial convolution and activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        # Main ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final processing
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        
        return x


class KinshipVerificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = IResNet(input_size=112)  # Input size fixed to match training
        
        # Try to load the pretrained weights
        try:
            pretrained_path = '/cephyr/users/mehdiyev/Alvis/kinship_project/src/pretrained_models/cosface_backbone_r50.pth'
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from {pretrained_path}")
                checkpoint = torch.load(pretrained_path)

                # Clean and load weights that match
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Remove unnecessary prefixes
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace('backbone.', '').replace('module.', '')
                    cleaned_state_dict[k] = v

                # Analyze architecture mismatch
                missing_keys_count, extra_keys_count, shape_mismatches_count = analyze_architecture_mismatch(
                    self.backbone.state_dict(),
                    cleaned_state_dict
                )

                # Load weights
                load_result = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
                print(f"Missing keys after loading: {len(load_result.missing_keys)}")
                if load_result.missing_keys:
                    print("First few missing keys:", load_result.missing_keys[:5])
                print("Successfully loaded pretrained weights")

            else:
                print(f"Warning: Pretrained weights not found at {pretrained_path}")
                print("Initializing with random weights")

        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            print("Initializing with random weights")
        
        # Small embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(512, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
            nn.ReLU(inplace=True)
        )
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)

class TripletKinshipLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    def forward(self, anchor, positive, negative):
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

class EarlyStopping:
    """Early stopping handler with proper metric tracking"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model):
        """
        Returns True if training should stop
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Current model state
        """
        if val_loss < float('inf') and val_loss != float('-inf'):
            if val_loss < self.best_loss - self.min_delta:
                # Loss improved
                self.best_loss = val_loss
                self.best_state = copy.deepcopy(model.state_dict())
                self.counter = 0
            else:
                # Loss didn't improve enough
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            print(f"Warning: Invalid loss value encountered: {val_loss}")
        
        return self.early_stop
    
    def get_best_state(self):
        """Returns the best model state"""
        return self.best_state

class KinshipDataset(Dataset):
    def __init__(self, csv_path, config, train=True):
        self.data = pd.read_csv(csv_path)
        self.config = config
        self.processor = ImageProcessor(train_mode=train)
        
        # Pre-validate all paths once and store valid triplets only
        print("Validating image paths...")
        valid_triplets = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            paths = [row['Anchor'], row['Positive'], row['Negative']]
            # if all(os.path.exists(p) for p in paths):
            valid_triplets.append(paths)
        
        self.triplets = valid_triplets
        print(f"Found {len(self.triplets)} valid triplets out of {len(self.data)}")
        
        # Optional: Cache most frequent images
        self.image_cache = {}
        if config.use_cache:  # Add this to Config class
            print("Building image cache...")
            all_paths = [path for triplet in self.triplets for path in triplet]
            path_counts = pd.Series(all_paths).value_counts()
            frequent_paths = path_counts[path_counts > 10].index  # Cache images used >10 times
            
            for path in tqdm(frequent_paths):
                try:
                    self.image_cache[path] = self.processor.process_face(path)
                except Exception as e:
                    print(f"Failed to cache {path}: {e}")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """Get item with proper error handling"""
        max_attempts = 3
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                new_idx = (idx + attempt) % len(self)
                paths = self.triplets[new_idx]
                
                images = []
                for path in paths:
                    if path in self.image_cache:
                        img = self.image_cache[path]
                    else:
                        img = self.processor.process_face(path)
                    
                    if img is None:
                        raise ValueError(f"Failed to load image: {path}")
                    
                    # Verify tensor is valid
                    if torch.isnan(img).any():
                        raise ValueError(f"NaN values in processed image: {path}")
                    
                    images.append(img)
                
                # Stack images into tensors
                return {
                    'anchor': images[0],
                    'positive': images[1],
                    'negative': images[2],
                    'paths': paths  # Include paths for debugging
                }
                
            except Exception as e:
                last_exception = e
                print(f"Error in __getitem__ (attempt {attempt+1}/{max_attempts}) for index {new_idx}: {str(e)}")
                continue
        
        # If we get here, all attempts failed
        raise RuntimeError(f"Failed to load valid triplet after {max_attempts} attempts. Last error: {str(last_exception)}")

def train_epoch(model, train_loader, criterion, optimizer, scaler, config, epoch, global_step):
    model.train()
    epoch_loss = 0
    total_accuracy = 0
    batch_distances = defaultdict(list)
    
    pbar = tqdm(train_loader, desc='Training')
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move data to device
            anchor = batch['anchor'].to(config.device, non_blocking=True)
            positive = batch['positive'].to(config.device, non_blocking=True)
            negative = batch['negative'].to(config.device, non_blocking=True)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negative_emb = model(negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Calculate distances and metrics
            with torch.no_grad():
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb)
                accuracy = (pos_dist < neg_dist).float().mean().item()
                margin_violations = torch.relu(pos_dist - neg_dist + config.margin).mean().item()
                
                # Store distances for distribution plotting
                if batch_idx % config.log_dist_freq == 0:  # Only store periodically
                    batch_distances['positive'] = pos_dist.cpu().numpy().tolist()
                    batch_distances['negative'] = neg_dist.cpu().numpy().tolist()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Increment global step correctly
                global_step = global_step + 1
                
                # Log metrics
                if batch_idx % config.log_batch_freq == 0:
                    curr_loss = loss.item() * config.gradient_accumulation_steps
                    wandb.log({
                        'batch/loss': curr_loss,
                        'batch/accuracy': accuracy,
                        'batch/margin_violations': margin_violations,
                        'batch/learning_rate': optimizer.param_groups[0]['lr'],
                        'batch/pos_dist_mean': pos_dist.mean().item(),
                        'batch/neg_dist_mean': neg_dist.mean().item()
                    }, step=global_step)
                    
                    # Log distance distributions
                    if len(batch_distances['positive']) > 0:
                        log_wandb_distance_distributions(
                            torch.tensor(batch_distances['positive']), 
                            torch.tensor(batch_distances['negative']), 
                            global_step
                        )
                        batch_distances = defaultdict(list)  # Clear after logging
            
            # Update metrics (without double counting)
            epoch_loss += loss.item()  # Already divided by accumulation steps
            total_accuracy += accuracy
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'accuracy': f'{total_accuracy/(batch_idx+1):.4f}',
                'margin_violations': f'{margin_violations:.4f}'
            })
            
            # Clear cache periodically
            if batch_idx % config.empty_cache_freq == 0:
                torch.cuda.empty_cache()
                
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\nCUDA out of memory. Skipping batch {batch_idx}")
            continue
    
    # Calculate final metrics
    metrics = {
        'train_loss': epoch_loss / (num_batches // config.gradient_accumulation_steps),
        'train_accuracy': total_accuracy / num_batches
    }
    
    return metrics, global_step

def validate(model, val_loader, criterion, config, epoch, global_step):
    """Validation function with proper metric calculation"""
    model.eval()
    val_loss = 0
    val_accuracy = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Move data to device
            anchor = batch['anchor'].to(config.device)
            positive = batch['positive'].to(config.device)
            negative = batch['negative'].to(config.device)
            
            # Forward pass
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            # Ensure embeddings are normalized
            anchor_out = F.normalize(anchor_out, p=2, dim=1)
            positive_out = F.normalize(positive_out, p=2, dim=1)
            negative_out = F.normalize(negative_out, p=2, dim=1)
            
            # Calculate distances
            pos_dist = F.pairwise_distance(anchor_out, positive_out)
            neg_dist = F.pairwise_distance(anchor_out, negative_out)
            
            # Calculate similarity scores (negative distance makes larger values = more similar)
            similarity_scores = torch.cat([-pos_dist, -neg_dist])
            labels = torch.cat([torch.ones_like(pos_dist), torch.zeros_like(neg_dist)])
            
            # Store predictions and labels
            all_predictions.extend(similarity_scores.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            # Calculate loss
            loss = criterion(anchor_out, positive_out, negative_out)
            val_loss += loss.item()
            
            # Calculate accuracy
            correct = (pos_dist < neg_dist).float().mean()
            val_accuracy += correct.item()
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    num_batches = len(val_loader)
    val_metrics = {
        'val_loss': val_loss / num_batches,
        'val_accuracy': val_accuracy / num_batches
    }
    
    # Calculate additional metrics if we have predictions
    if len(all_predictions) > 1:
        # Calculate F1 score
        pred_labels = (all_predictions > 0).astype(float)
        val_metrics['val_f1'] = f1_score(all_labels, pred_labels)
        
        # Calculate precision and recall
        val_metrics['val_precision'] = precision_score(all_labels, pred_labels)
        val_metrics['val_recall'] = recall_score(all_labels, pred_labels)
        
        try:
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(all_labels, all_predictions)
            val_metrics['val_roc_auc'] = auc(fpr, tpr)
            
            # Create ROC curve for wandb
            val_metrics['val_roc_curve'] = wandb.plot.roc_curve(
                all_labels,
                all_predictions,
                labels=['Non-kin', 'Kin']
            )
        except Exception as e:
            print(f"Warning: Could not calculate ROC curve: {str(e)}")
    
    # Log metrics
    wandb.log(val_metrics, step=global_step)
    
    return val_metrics
def main():
    """Enhanced main training loop with improved logging and visualization"""
    config = Config()
    global_step = 0
    
    try:
        # Initialize wandb with proper configuration
        wandb.init(
            project="kinship-verification-18nov",
            name=config.experiment_name,
            config=vars(config),
            dir=config.log_dir,
            resume="allow"  # Allow resuming interrupted runs
        )
        
        # Define custom step metrics before any logging
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")
        wandb.define_metric("batch/*", step_metric="global_step")
        wandb.define_metric("image_processing/*", step_metric="global_step")
        
        print(f"\nExperiment directory: {config.experiment_dir}")
        print(f"Using device: {config.device}")
        
        # Initialize datasets and dataloaders
        train_dataset = KinshipDataset(config.train_path, config, train=True)
        val_dataset = KinshipDataset(config.val_path, config, train=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        # Model initialization
        model = KinshipVerificationModel(config).to(config.device)
        scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
        criterion = TripletKinshipLoss(margin=config.margin)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.min_lr
        )
        
        early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            # Train with global step tracking
            train_metrics, global_step = train_epoch(
                model, train_loader, criterion, optimizer, 
                scaler,  # Add this
                config, epoch, global_step
            )
            
            # Validate with global step
            val_metrics = validate(
                model, val_loader, criterion, 
                config, epoch, global_step
            )
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics with synchronized global step
            wandb.log({
                'epoch': epoch,
                'global_step': global_step,
                'learning_rate': current_lr,
                'train/loss': train_metrics['train_loss'],
                'train/accuracy': train_metrics['train_accuracy'],
                'val/loss': val_metrics['val_loss'],
                'val/accuracy': val_metrics['val_accuracy']
            }, step=global_step)
            
            # Log embeddings periodically with global step
            if epoch % config.log_embeddings_freq == 0:
                log_wandb_embeddings(model, val_loader, config, global_step)
            
            # Save best model
            if val_metrics.get('val_f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('val_f1', 0)
                best_model_state = copy.deepcopy(model.state_dict())
                
                wandb.run.summary.update({
                    "best_val_f1": best_val_f1,
                    "best_epoch": epoch,
                    "best_global_step": global_step
                })
                
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_metrics, train_metrics, best_val_f1, config
                )
                
                print(f"\nNew best model saved! F1: {best_val_f1:.4f}")
            
            # Early stopping check
            if early_stopping(val_metrics['val_loss'], model):
                print("\nEarly stopping triggered!")
                wandb.run.summary.update({
                    "stopped_epoch": epoch,
                    "stopped_global_step": global_step
                })
                break
            
            print_metrics(train_metrics, val_metrics, current_lr)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        wandb.run.summary.update({
            "interrupted_epoch": epoch if 'epoch' in locals() else 0,
            "interrupted_global_step": global_step
        })
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        wandb.run.summary["error"] = str(e)
        raise
    finally:
        cleanup_and_log_final_metrics(
            model, val_loader, criterion, config,
            best_model_state, best_val_f1, global_step
        )
        wandb.finish()

if __name__ == '__main__':
    main()