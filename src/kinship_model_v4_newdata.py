import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from pathlib import Path
import json
import time
from torch.cuda.amp import autocast, GradScaler


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class Config:
    def __init__(self):
        # Data paths
        self.train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/train_triplets_enhanced.csv'
        self.val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/val_triplets_enhanced.csv'
        self.test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/test_triplets_enhanced.csv'
        self.output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model'
        
        # Training settings
        self.batch_size = 64
        self.num_epochs = 10
        self.warmup_epochs = 3  # Increased warmup
        self.learning_rate = 1e-5
        self.weight_decay = 5e-4
        self.min_lr = 1e-6
        self.dropout_rate = 0.5  # Increased dropout
        self.num_workers = min(16, os.cpu_count())
        
        # Learning rate schedule settings
        self.pct_start = 0.3          # Percentage of training for warmup
        self.div_factor = 25          # Initial LR division factor
        self.final_div_factor = 1e4   # Final LR division factor
        
        # Model settings
        self.embedding_dim = 512  # Using full embedding dimension
        self.margin = 0.5  # Increased margin        not trained yet
        
        # Device settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # New settings
        self.exp_name = f'kinship_model_v4n_hand_27nov'
        self.save_every = 1
        self.eval_every = 100
        self.pin_memory = True
        self.early_stopping_patience = 5
        
        # Create output directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create output directories for experiments"""
        self.exp_dir = Path(self.output_dir) / self.exp_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        
        # Create directories
        for dir_path in [self.exp_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            config_dict = {k: str(v) for k, v in vars(self).items() 
                          if not k.startswith('_') and isinstance(v, (str, int, float, bool))}
            json.dump(config_dict, f, indent=4)

class KinshipDataset(Dataset):
    """Improved Dataset class with better error handling and validation"""
    def __init__(self, csv_path, transform=None, validation=False):
        self.data = pd.read_csv(csv_path)
        self.transform = transform or self.get_default_transforms(validation)
        self.validation = validation
        
        # Validate and filter data
        self.validate_data()
    
    @staticmethod
    def get_default_transforms(validation):
        """Get default transforms based on training/validation mode - specialized for face images"""
        if validation:
            return transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),  # Faces are horizontally symmetric
                # Mild color jittering for lighting variations
                transforms.ColorJitter(
                    brightness=0.1,  # Lighting changes
                    contrast=0.1,    # Contrast variations
                    saturation=0.05,  # Mild saturation changes
                ),
                # Very mild affine transformations
                transforms.RandomAffine(
                    degrees=(-5, 5),        # Subtle rotation
                    translate=(0.05, 0.05),  # Minor translations
                    scale=(0.95, 1.05)      # Subtle scaling
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # Mild random erasing for occlusion robustness
                transforms.RandomErasing(
                    p=0.1,           # Lower probability
                    scale=(0.02, 0.1),  # Smaller regions
                    value=0
                )
            ])
    
    def validate_data(self):
        """Validate and filter the dataset"""
        valid_rows = []
        print(f"Validating {len(self.data)} image triplets...")
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            if self.validate_triplet(row):
                valid_rows.append(row)
        
        self.data = pd.DataFrame(valid_rows)
        print(f"Found {len(self.data)} valid triplets")
    
    def validate_triplet(self, row):
        """Validate a single triplet of images"""
        paths = [row['Anchor'], row['Positive'], row['Negative']]
        return all(os.path.exists(p) for p in paths)
    
    def load_image(self, path):
        """Load and transform an image with error handling"""
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load images
        anchor = self.load_image(row['Anchor'])
        positive = self.load_image(row['Positive'])
        negative = self.load_image(row['Negative'])
        
        # Check for failed loads
        if any(img is None for img in [anchor, positive, negative]):
            # Return a different triplet
            print("Error loading images, returning a different triplet")
            return self.__getitem__((idx + 1) % len(self))
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'paths': [row['Anchor'], row['Positive'], row['Negative']]
        }

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

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
    """Improved ResNet architecture with additional regularization"""
    def __init__(self, config, input_size=112):
        super().__init__()
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        
        # Layer configurations for ResNet-50
        blocks = [3, 4, 14, 3]
        
        # Add dropout layers
        self.dropout = nn.Dropout(p=0.2)
        self.layer1_dropout = nn.Dropout(p=0.05)
        self.layer2_dropout = nn.Dropout(p=0.1)
        self.layer3_dropout = nn.Dropout(p=0.15)
        self.layer4_dropout = nn.Dropout(p=0.15)
        
        # Main layers with stride=2 for downsampling
        self.layer1 = self._make_layer(IRBlock, 64, 64, blocks[0], stride=2)
        self.layer2 = self._make_layer(IRBlock, 64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(IRBlock, 128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(IRBlock, 256, 512, blocks[3], stride=2)
        
        # Final layers
        self.bn2 = nn.BatchNorm2d(512)
        
        # Calculate feature size: input_size/(2^4) for 4 stride-2 layers
        self.feature_size = (input_size // 16) ** 2 * 512
        self.fc = nn.Linear(self.feature_size, 512)
        self.features = nn.BatchNorm1d(512)
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(512)

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
        """Initialize model weights with improved initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass with additional regularization"""
        # Initial convolution and activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        # Main ResNet blocks with dropout
        x = self.layer1(x)
        x = self.layer1_dropout(x)
        
        x = self.layer2(x)
        x = self.layer2_dropout(x)
        
        x = self.layer3(x)
        x = self.layer3_dropout(x)
        
        x = self.layer4(x)
        x = self.layer4_dropout(x)
        
        # Final processing
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        x = self.layer_norm(x)  
        x = F.normalize(x, p=2, dim=1)
        
        return x

class KinshipVerificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = IResNet(config,input_size=112)  # Input size fixed to match training
        
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
        
        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
        
        # Small embedding layer
        self.embedding = nn.Identity() 
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)

class TripletKinshipLoss(nn.Module):
    def __init__(self, margin=0.3, mining='semi-hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(self, anchor, positive, negative):
        # Remove normalization here since inputs are already normalized from model
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        if self.mining == 'semi-hard':
            mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            masked_neg_dist = neg_dist * mask.float() + (1 - mask.float()) * neg_dist.detach()
            loss = F.relu(pos_dist - masked_neg_dist + self.margin)
        else:
            loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Store statistics
        self.pos_dist_mean = pos_dist.mean().item()
        self.neg_dist_mean = neg_dist.mean().item()
        self.hard_triplets_ratio = mask.float().mean().item() if self.mining == 'semi-hard' else 1.0
        
        return loss.mean()

def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Save regular checkpoint
    checkpoint_path = config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if needed
    if is_best:
        best_path = config.checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch):
    model.train()
    scaler = GradScaler()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
    for batch_idx, batch in enumerate(pbar):
        anchor = batch['anchor'].to(config.device, non_blocking=True)
        positive = batch['positive'].to(config.device, non_blocking=True)
        negative = batch['negative'].to(config.device, non_blocking=True)
        
        batch_size = anchor.size(0)
        num_samples += batch_size
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            loss = criterion(anchor_embed, positive_embed, negative_embed)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        with torch.no_grad():
            pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
            neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
            correct = (pos_dist < neg_dist).float().sum()
            
            running_loss += loss.item() * batch_size
            running_acc += correct.item()
        
        avg_loss = running_loss / num_samples
        avg_acc = running_acc / num_samples
        
        pbar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'acc': f"{avg_acc:.4f}",
            'pos_d': f"{pos_dist.mean().item():.4f}",
            'neg_d': f"{neg_dist.mean().item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': avg_acc
    }


def validate(model, val_loader, criterion, config):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            anchor = batch['anchor'].to(config.device, non_blocking=True)
            positive = batch['positive'].to(config.device, non_blocking=True)
            negative = batch['negative'].to(config.device, non_blocking=True)
            
            with autocast():
                anchor_embed = model(anchor)
                positive_embed = model(positive)
                negative_embed = model(negative)
                
                pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
                neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
                loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            batch_size = anchor.size(0)
            num_samples += batch_size
            
            # Calculate accuracy
            correct = (pos_dist < neg_dist).float().sum()
            
            running_loss += loss.item() * batch_size
            running_acc += correct.item()
            
    val_loss = running_loss / num_samples
    val_acc = running_acc / num_samples
    
    metrics = {
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }
    
    return metrics

def test_model(model, test_loader, config):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    all_pos_dists = []
    all_neg_dists = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            anchor = batch['anchor'].to(config.device, non_blocking=True)
            positive = batch['positive'].to(config.device, non_blocking=True)
            negative = batch['negative'].to(config.device, non_blocking=True)
            
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            
            pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
            neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
            
            correct += (pos_dist < neg_dist).sum().item()
            total += anchor.size(0)
            
            all_pos_dists.extend(pos_dist.cpu().numpy())
            all_neg_dists.extend(neg_dist.cpu().numpy())
    
    accuracy = correct / total
    auc = calculate_auc(all_pos_dists, all_neg_dists)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Mean Positive Distance: {np.mean(all_pos_dists):.4f}")
    print(f"Mean Negative Distance: {np.mean(all_neg_dists):.4f}")
    
    return accuracy, auc

def calculate_auc(pos_dists, neg_dists):
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    y_score = np.concatenate([-np.array(pos_dists), -np.array(neg_dists)])
    return roc_auc_score(y_true, y_score)


def main():
    # Initialize config and directories
    config = Config()
    
    # Set seed
    set_seed(17)
    
    # Initialize wandb
    wandb.init(
        project="kinship-verification-v4",
        name=config.exp_name,
        config=vars(config),
        dir=str(config.log_dir)
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create datasets and dataloaders
    train_dataset = KinshipDataset(config.train_path, validation=False)
    val_dataset = KinshipDataset(config.val_path, validation=True)
    test_dataset = KinshipDataset(config.test_path, validation=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        prefetch_factor=2, 
        persistent_workers=True  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Initialize model, criterion, optimizer
    model = KinshipVerificationModel(config)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model = model.to(config.device)
    criterion = TripletKinshipLoss(margin=config.margin)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  
        div_factor=10,
        final_div_factor=1e3
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, config, epoch
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, config
        )
        
        
        # Update best model
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
        
        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {**train_metrics, **val_metrics},
                config, is_best
            )
        
        # Log metrics
        wandb.log({
            **train_metrics,
            **val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        })
        
        # Print metrics
        print(f"Train Loss: {train_metrics['train_loss']:.4f} "
              f"Train Acc: {train_metrics['train_accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f} "
              f"Val Acc: {val_metrics['val_accuracy']:.4f}")
        
        # Early stopping check
        if early_stopping(val_metrics['val_loss']):
            print("Early stopping triggered!")
            break
    
    # Final test evaluation
    print("\nRunning final evaluation on test set...")
    test_acc, test_auc = test_model(model, test_loader, config)
    
    # Log final results
    wandb.log({
        'test_accuracy': test_acc,
        'test_auc': test_auc
    })
    
    
    wandb.finish()

if __name__ == "__main__":
    main()