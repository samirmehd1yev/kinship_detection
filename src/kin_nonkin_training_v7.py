import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import math



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(17)
np.random.seed(17)
random.seed(17)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(17)

# Define output directories using the provided path
output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model_v7_2'
model_dir = os.path.join(output_dir, 'model')
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
stats_dir = os.path.join(output_dir, 'stats')
plot_dir = os.path.join(output_dir, 'plots')

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def calculate_dataset_statistics(csv_path, config):
    """Calculate mean and std from all dataset images"""
    stats_file = os.path.join(stats_dir, 'dataset_stats.npz')
    if os.path.exists(stats_file):
        stats = np.load(stats_file)
        return stats['mean'], stats['std']
        
    data = pd.read_csv(csv_path)
    # Combine all image paths
    image_paths = []
    for _, row in data.iterrows():
        image_paths.extend([row['Anchor'], row['Positive'], row['Negative']])
    
    # Remove duplicates
    image_paths = list(set(image_paths))
    
    # Initialize arrays for mean and std calculation
    mean = np.zeros(3)
    std = np.zeros(3)
    
    # Basic transforms for consistent size
    basic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor()
    ])
    
    print(f"Calculating dataset statistics using {len(image_paths)} images...")
    n_images = 0
    
    # First pass: mean
    for img_path in tqdm(image_paths, desc="Calculating mean"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = basic_transform(img)
            mean += torch.mean(img, dim=(1,2)).numpy()
            n_images += 1
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    mean = mean / n_images
    print(f"Calculated mean using {n_images} valid images")
    
    # Second pass: std
    n_images = 0  # Reset counter for std calculation
    for img_path in tqdm(image_paths, desc="Calculating std"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = basic_transform(img)
            std += torch.mean((img - torch.tensor(mean).view(3,1,1))**2, dim=(1,2)).numpy()
            n_images += 1
        except Exception as e:
            continue
    
    std = np.sqrt(std / n_images)
    print(f"Calculated std using {n_images} valid images")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    # Save statistics
    np.savez(stats_file, mean=mean, std=std)
    return mean, std

def save_checkpoint(state, is_best, epoch=None):
    """Save model checkpoint"""
    if epoch is not None:
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
    else:
        checkpoint_path = os.path.join(model_dir, 'last_checkpoint.pth')
    
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(model_dir, 'best_model.pth')
        torch.save(state, best_path)

def load_checkpoint(model, path, device):
    """Load checkpoint with device handling"""
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return training state
        return {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('val_loss', float('inf')),
            'feature_optimizer_state': checkpoint.get('feature_optimizer_state_dict'),
            'main_optimizer_state': checkpoint.get('main_optimizer_state_dict'),
            'feature_scheduler_state': checkpoint.get('feature_scheduler_state_dict'),
            'main_scheduler_state': checkpoint.get('main_scheduler_state_dict')
        }
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None

def save_training_plot(train_losses, val_losses):
    """Save training history plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'training_history.png'))
    plt.close()

def save_results(results):
    """Save evaluation results"""
    results_file = os.path.join(stats_dir, 'test_results.npz')
    np.savez(results_file,
             predictions=results['predictions'],
             labels=results['labels'])
    
    # Save precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(results['recall'], results['precision'], 
             label=f'AP={results["average_precision"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'precision_recall_curve.png'))
    plt.close()
class KinshipConfig:
    def __init__(self):
        self.input_size = 224
        self.face_embedding_size = 512
        self.batch_size = 128
        self.learning_rate = 2e-4
        self.weight_decay = 1e-3
        self.num_epochs = 15
        self.warmup_epochs = 2
        self.dropout_rate = 0.4
        self.train_path = '../data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
        self.val_path = '../data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
        self.test_path = '../data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.base = InceptionResnetV1(pretrained='vggface2')
        
        # Define SE blocks with correct channel sizes based on the InceptionResNetV1 architecture
        self.se_blocks = nn.ModuleList([
            SEBlock(256),    # After conv2d_4b (before repeat_1)
            SEBlock(896),    # After mixed_6a (between repeat_1 and repeat_2)
            SEBlock(1792),   # After mixed_7a (between repeat_2 and repeat_3)
            SEBlock(1792)    # After block8
        ])
        
        self.projection = nn.Sequential(
            nn.Linear(1792, embedding_size),
            nn.LayerNorm(embedding_size)
        )
        
        # Freeze early layers
        for param in list(self.base.parameters())[:-20]:
            param.requires_grad = False

    def forward(self, x):
        # Initial blocks (3 -> 32 -> 32 -> 64 -> 80 -> 192 -> 256)
        x = self.base.conv2d_1a(x)  # 3 -> 32
        x = self.base.conv2d_2a(x)  # 32 -> 32
        x = self.base.conv2d_2b(x)  # 32 -> 64
        x = self.base.maxpool_3a(x)
        x = self.base.conv2d_3b(x)  # 64 -> 80
        x = self.base.conv2d_4a(x)  # 80 -> 192
        x = self.base.conv2d_4b(x)  # 192 -> 256
        x = self.se_blocks[0](x)    # SE on 256 channels
        
        # Inception-ResNet-A modules (256 channels)
        x = self.base.repeat_1(x)   # 5x Block35 (keeps 256 channels)
        
        # Reduction-A (256 -> 896)
        x = self.base.mixed_6a(x)   # 256 -> 896
        x = self.se_blocks[1](x)    # SE on 896 channels
        
        # Inception-ResNet-B modules (896 channels)
        x = self.base.repeat_2(x)   # 10x Block17 (keeps 896 channels)
        
        # Reduction-B (896 -> 1792)
        x = self.base.mixed_7a(x)   # 896 -> 1792
        x = self.se_blocks[2](x)    # SE on 1792 channels
        
        # Inception-ResNet-C modules (1792 channels)
        x = self.base.repeat_3(x)   # 5x Block8 (keeps 1792 channels)
        x = self.base.block8(x)     # Final Block8 with no ReLU
        x = self.se_blocks[3](x)    # SE on final 1792 channels
        
        # Final processing
        x = self.base.avgpool_1a(x)
        x = self.base.dropout(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        
        return F.normalize(x, p=2, dim=1)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction_ratio, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.reduced_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        assert c == self.in_channels, f"Expected {self.in_channels} channels, got {c}"
        
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        
    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        
        # Layer norm first
        x1 = self.layernorm1(x1)
        x2 = self.layernorm2(x2)
        
        # Project to queries, keys, and values
        q = self.query(x1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Combine values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final projection
        out = self.proj_dropout(self.proj(out))
        return out.squeeze(1)  # Remove sequence dimension

class ImageProcessor:
    def __init__(self, config, is_training=False):
        self.config = config
        self.is_training = is_training
        stats_file = os.path.join(stats_dir, 'dataset_stats.npz')  
        if os.path.exists(stats_file):
            stats = np.load(stats_file)
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            self.mean, self.std = calculate_dataset_statistics(config.train_path, config) 
            
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def process_face(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.is_training:
                img = self.train_transform(img)
            else:
                img = self.val_transform(img)
            return img
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None

class KinshipDataset(Dataset):
    def __init__(self, csv_path, config, is_training=False):
        self.data = pd.read_csv(csv_path)
        self.config = config
        self.processor = ImageProcessor(config, is_training)
        pos_pairs = []
        neg_pairs = []
        
        for _, row in self.data.iterrows():
            pos_pairs.append((row['Anchor'], row['Positive'], 1))
            neg_pairs.append((row['Anchor'], row['Negative'], 0))
        
        min_pairs = min(len(pos_pairs), len(neg_pairs))
        if is_training:
            pos_pairs = pos_pairs[:min_pairs]
            neg_pairs = neg_pairs[:min_pairs]
            
        self.pairs = pos_pairs + neg_pairs
        if is_training:
            random.shuffle(self.pairs)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_path, other_path, is_related = self.pairs[idx]
        anchor = self.processor.process_face(anchor_path)
        other = self.processor.process_face(other_path)
        
        if anchor is None or other is None:
            return self.__getitem__((idx + 1) % len(self))
        
        # Convert tensors but don't move to device here
        return {
            'anchor': anchor,
            'other': other,
            'is_related': torch.tensor(is_related, dtype=torch.float)
        }
    
class KinshipVerifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Modified feature extractor
        self.feature_extractor = EnhancedFeatureExtractor(config.face_embedding_size)
        
        # 2. Simpler attention mechanism
        self.cross_attention = MultiHeadCrossAttention(
            config.face_embedding_size,
            num_heads=8,
            dropout=0.2
        )
        
        # 3. Simplified fusion network
        fusion_size = config.face_embedding_size * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, fusion_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(fusion_size // 2),
        )
        
        # 4. Simple classifier head
        self.classifier = nn.Linear(fusion_size // 2, 1)
        
    def forward(self, anchor, other):
        device = next(self.parameters()).device
        anchor = anchor.to(device)
        other = other.to(device)
        
        # Extract and normalize features
        anchor_features = self.feature_extractor(anchor)
        other_features = self.feature_extractor(other)
        
        # Single cross-attention layer
        attended_anchor = self.cross_attention(anchor_features, other_features)
        attended_other = self.cross_attention(other_features, anchor_features)
        
        # Combine features
        combined = torch.cat([attended_anchor, attended_other], dim=1)
        fused = self.fusion(combined)
        score = self.classifier(fused)
        
        return {
            'kinship_score': score.squeeze(),
            'anchor_features': anchor_features,
            'other_features': other_features,
            'fused_features': fused
        }
              
class HybridLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.config = config
        
    def forward(self, predictions, targets):
        device = predictions['kinship_score'].device
        is_related = targets['is_related'].to(device)
        
        # BCE Loss
        bce_loss = self.bce_loss(predictions['kinship_score'], is_related)
        
        # Triplet Loss
        pos_mask = is_related == 1
        neg_mask = is_related == 0
        
        if torch.sum(pos_mask) > 0 and torch.sum(neg_mask) > 0:
            anchor_embeds = predictions['anchor_features']
            other_embeds = predictions['other_features']
            
            pos_embeds = other_embeds[pos_mask]
            neg_embeds = other_embeds[neg_mask]
            
            # Use only valid triplets
            num_triplets = min(pos_embeds.size(0), neg_embeds.size(0))
            if num_triplets > 0:
                triplet_loss = self.triplet_loss(
                    anchor_embeds[:num_triplets],
                    pos_embeds[:num_triplets],
                    neg_embeds[:num_triplets]
                )
                return bce_loss + 0.5 * triplet_loss
                
        return bce_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0001):
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

def train_epoch(model, train_loader, optimizers, criterion, device, scaler):
    """Training epoch function"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    feature_optimizer, main_optimizer = optimizers
    progress_bar = tqdm(train_loader, desc='Training')

    for batch in progress_bar:
        # Move data to device
        anchor = batch['anchor'].to(device)
        other = batch['other'].to(device)
        is_related = batch['is_related'].to(device)
        
        # Create device-aware batch
        device_batch = {
            'anchor': anchor,
            'other': other,
            'is_related': is_related
        }
        
        # Zero gradients
        feature_optimizer.zero_grad()
        main_optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            predictions = model(anchor, other)
            loss = criterion(predictions, device_batch)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step with scaler
        scaler.step(feature_optimizer)
        scaler.step(main_optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Compute accuracy
        pred_labels = (torch.sigmoid(predictions['kinship_score']) > 0.5).float()
        total_correct += (pred_labels == is_related).sum().item()
        total_samples += is_related.size(0)

        # Update progress bar
        avg_loss = total_loss / (progress_bar.n + 1)
        avg_accuracy = total_correct / total_samples
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}', 
            'acc': f'{avg_accuracy:.4f}'
        })

    return total_loss / len(train_loader), total_correct / total_samples

# Helper function for validation
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            
            device_batch = {
                'anchor': anchor,
                'other': other,
                'is_related': is_related
            }
            
            # Forward pass
            predictions = model(anchor, other)
            loss = criterion(predictions, device_batch)
            
            # Compute metrics
            total_loss += loss.item()
            pred_labels = (torch.sigmoid(predictions['kinship_score']) > 0.5).float()
            total_correct += (pred_labels == is_related).sum().item()
            total_samples += is_related.size(0)
    
    return total_loss / len(val_loader), total_correct / total_samples

def save_checkpoint(state, is_best, epoch=None):
    """Save model checkpoint with improved handling"""
    # Save latest checkpoint
    if epoch is not None:
        checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
    else:
        checkpoint_path = os.path.join(model_dir, 'last_checkpoint.pth')
    
    torch.save(state, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(model_dir, 'best_model.pth')
        torch.save(state, best_path)
        
def load_checkpoint(model, path, device):
    """Load checkpoint with device handling"""
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Return training state
        return {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('val_loss', float('inf')),
            'feature_optimizer_state': checkpoint.get('feature_optimizer_state_dict'),
            'main_optimizer_state': checkpoint.get('main_optimizer_state_dict'),
            'feature_scheduler_state': checkpoint.get('feature_scheduler_state_dict'),
            'main_scheduler_state': checkpoint.get('main_scheduler_state_dict')
        }
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None

def train_model(model, train_loader, val_loader, config, resume_training=True):
    device = next(model.parameters()).device
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Try to load the best model checkpoint if it exists
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    last_checkpoint_path = os.path.join(model_dir, 'last_checkpoint.pth')
    
    if resume_training:
        # First try to load the last checkpoint
        if os.path.exists(last_checkpoint_path):
            print("Resuming from last checkpoint...")
            checkpoint_state = load_checkpoint(model, last_checkpoint_path, device)
        # If last checkpoint doesn't exist, try the best model
        elif os.path.exists(best_model_path):
            print("Resuming from best model checkpoint...")
            checkpoint_state = load_checkpoint(model, best_model_path, device)
        else:
            print("No checkpoint found, starting from scratch...")
            checkpoint_state = None
            
        if checkpoint_state is not None:
            start_epoch = checkpoint_state['epoch'] + 1
            best_val_loss = checkpoint_state['best_val_loss']
            print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
    
    # Initialize optimizers
    feature_optimizer = torch.optim.AdamW(
        model.feature_extractor.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    main_optimizer = torch.optim.AdamW([
        {'params': model.cross_attention.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=2e-4, weight_decay=0.01)
    
    # Load optimizer states if resuming
    if resume_training and checkpoint_state is not None:
        if checkpoint_state['feature_optimizer_state']:
            feature_optimizer.load_state_dict(checkpoint_state['feature_optimizer_state'])
        if checkpoint_state['main_optimizer_state']:
            main_optimizer.load_state_dict(checkpoint_state['main_optimizer_state'])
    
    # Calculate remaining steps for scheduler
    remaining_epochs = config.num_epochs - start_epoch
    num_warmup_steps = len(train_loader) * 2  # 2 epochs of warmup
    num_training_steps = len(train_loader) * remaining_epochs
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    # Initialize schedulers
    feature_scheduler = torch.optim.lr_scheduler.LambdaLR(feature_optimizer, lr_lambda)
    main_scheduler = torch.optim.lr_scheduler.LambdaLR(main_optimizer, lr_lambda)
    
    # Load scheduler states if resuming
    if resume_training and checkpoint_state is not None:
        if checkpoint_state['feature_scheduler_state']:
            feature_scheduler.load_state_dict(checkpoint_state['feature_scheduler_state'])
        if checkpoint_state['main_scheduler_state']:
            main_scheduler.load_state_dict(checkpoint_state['main_scheduler_state'])
    
    criterion = HybridLoss(config)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(start_epoch, config.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            # Training
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            progress_bar = tqdm(train_loader, desc='Training')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Zero gradients
                feature_optimizer.zero_grad()
                main_optimizer.zero_grad()
                
                # Move data to device
                anchor = batch['anchor'].to(device)
                other = batch['other'].to(device)
                is_related = batch['is_related'].to(device)
                
                device_batch = {
                    'anchor': anchor,
                    'other': other,
                    'is_related': is_related
                }
                
                # Forward pass with mixed precision
                with autocast():
                    predictions = model(anchor, other)
                    loss = criterion(predictions, device_batch)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(feature_optimizer)
                scaler.step(main_optimizer)
                scaler.update()
                
                # Update learning rate
                feature_scheduler.step()
                main_scheduler.step()
                
                # Compute metrics
                total_loss += loss.item()
                pred_labels = (torch.sigmoid(predictions['kinship_score']) > 0.5).float()
                total_correct += (pred_labels == is_related).sum().item()
                total_samples += is_related.size(0)
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                avg_accuracy = total_correct / total_samples
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_accuracy:.4f}',
                    'lr': f'{main_optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_samples
            
            # Validation
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            print(f"Learning Rate: {main_optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_optimizer_state_dict': feature_optimizer.state_dict(),
                'main_optimizer_state_dict': main_optimizer.state_dict(),
                'feature_scheduler_state_dict': feature_scheduler.state_dict(),
                'main_scheduler_state_dict': main_scheduler.state_dict(),
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                
            save_checkpoint(checkpoint, is_best, epoch=epoch)
            save_training_plot(train_losses, val_losses)
            
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        save_checkpoint(checkpoint, False, epoch=epoch)
        print("Checkpoint saved. You can resume training later.")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Saving emergency checkpoint...")
        save_checkpoint(checkpoint, False, epoch=epoch)
        raise e
    
    return model, best_val_loss

def evaluate(model, test_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            
            predictions = model(anchor, other)
            pred_probs = torch.sigmoid(predictions['kinship_score'])
            pred_labels = (pred_probs > 0.5).float()
            
            total_correct += (pred_labels == is_related).sum().item()
            total_samples += is_related.size(0)
            
            all_predictions.extend(pred_probs.cpu().numpy())
            all_labels.extend(is_related.cpu().numpy())
    
    accuracy = total_correct / total_samples
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    roc_auc = roc_auc_score(all_labels, all_predictions)
    ap = average_precision_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'average_precision': ap,
        'predictions': all_predictions,
        'labels': all_labels,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    config = KinshipConfig()
    
    # Create datasets and dataloaders
    train_dataset = KinshipDataset(config.train_path, config, is_training=True)
    val_dataset = KinshipDataset(config.val_path, config, is_training=False)
    test_dataset = KinshipDataset(config.test_path, config, is_training=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = KinshipVerifier(config).to(device)
    
    # Train model with resume capability
    model, best_val_loss = train_model(model, train_loader, val_loader, config, resume_training=True)
    
    # Load best model for evaluation
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(model, best_model_path, device)
    
    # Evaluate and save results
    test_results = evaluate(model, test_loader)
    save_results(test_results)
    print("Evaluation complete. Results saved.")