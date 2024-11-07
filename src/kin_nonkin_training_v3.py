import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Directory setup
output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model3'
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

class KinshipConfig:
    def __init__(self):
        # Model architecture
        self.input_size = 112
        self.face_embedding_size = 512
        self.feature_dim = 512       # Changed from 2048
        self.embedding_size = 512  
        
        # ArcFace parameters
        self.arcface_scale = 16
        self.arcface_margin = 0.1
        self.easy_margin = True
        
        # Training settings
        self.batch_size = 32
        self.learning_rate = 1e-5
        self.weight_decay = 1e-4
        self.num_epochs = 10
        self.warmup_epochs = 5
        
        # Mixed precision training
        self.dtype = torch.float32
        self.use_amp = True
        
        # Optimizer settings
        self.beta1 = 0.9
        self.beta2 = 0.999
        
        
        # Learning rate scheduler
        self.min_lr = 1e-6
        self.T_0 = 10  # Number of iterations for the first restart
        self.T_mult = 2  # Factor to increase T_i after a restart
        
        # Data settings
        self.train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
        self.val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
        self.test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'

# Advanced image preprocessing
class ImageProcessor:
    @staticmethod
    def read_image(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def preprocess_face(img, target_size=112):
        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        
        # Standardize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Resize with aspect ratio preservation
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        delta_h = target_size - new_h
        delta_w = target_size - new_w
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return img

    
    @staticmethod
    def process_face(img_path, target_size=112):
        try:
            img = ImageProcessor.read_image(img_path)
            img = ImageProcessor.preprocess_face(img, target_size)
            # Convert to float32 explicitly
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            return img
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
        return None
            

# Dataset with augmentation
class KinshipDataset(Dataset):
    def __init__(self, csv_path, config, is_training=True):
        self.data = pd.read_csv(csv_path)
        self.config = config
        self.processor = ImageProcessor()
        self.is_training = is_training
        
        # Create pairs
        self.pairs = []
        for _, row in self.data.iterrows():
            self.pairs.append((row['Anchor'], row['Positive'], 1))
            self.pairs.append((row['Anchor'], row['Negative'], 0))
        
        print(f"Loaded {len(self.pairs)} pairs")
        print("\nKinship distribution:")
        kin_counts = pd.Series([pair[2] for pair in self.pairs]).value_counts()
        print(kin_counts)

    def __len__(self):
        return len(self.pairs)
    
    def augment_image(self, img):
        """Apply augmentation only during training"""
        if not self.is_training:
            return img
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = torch.flip(img, [2])
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            img = img * brightness_factor
            img = torch.clamp(img, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = torch.mean(img, dim=[1, 2], keepdim=True)
            img = (img - mean) * contrast_factor + mean
            img = torch.clamp(img, 0, 1)
            
        return img
    
    def __getitem__(self, idx):
        max_retries = 3
        current_try = 0
        
        while current_try < max_retries:
            anchor_path, other_path, is_related = self.pairs[idx]
            
            anchor = self.processor.process_face(anchor_path)
            other = self.processor.process_face(other_path)
            
            if anchor is not None and other is not None:
                if self.is_training:
                    anchor = self.augment_image(anchor)
                    other = self.augment_image(other)
                
                return {
                    'anchor': anchor.float(),
                    'other': other.float(),
                    'is_related': torch.tensor(is_related, dtype=torch.long)
                }
            
            current_try += 1
            idx = (idx + 1) % len(self)
        
        raise RuntimeError(f"Failed to load valid image pair after {max_retries} attempts")
# Feature Extractor with advanced architecture
class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        
        # Deep features extraction
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        
        # Global context block
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        
        # Final layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, config.face_embedding_size)
        self.bn_final = nn.BatchNorm1d(config.face_embedding_size)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        # Apply global context
        context = self.global_context(x)
        x = x * context
        
        # Global pooling and final layers
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_final(x)
        
        return F.normalize(x, p=2, dim=1)

# ArcFace Loss implementation
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=16.0, margin=0.1, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # Weight initialization with smaller values
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight, gain=0.1)  # Added gain parameter
        
    def forward(self, embeddings, labels):
        # Add numerical stability
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        
        cosine = F.linear(embeddings, weights)
        # Add clipping to prevent numerical instability
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-7)  # Added small epsilon
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert labels to one-hot with smoothing
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

# Advanced Kinship Model
class KinshipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Ensure model parameters are in float32
        self.to(torch.float32)
        
        # Feature extractor with shared weights
        self.feature_extractor = FeatureExtractor(config)
        
        # Additional layer to reduce concatenated feature dimension
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.face_embedding_size * 2, config.embedding_size),
            nn.BatchNorm1d(config.embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Add dropout
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.BatchNorm1d(config.embedding_size),
            nn.ReLU(inplace=True)
        )
        
        # ArcFace loss
        self.arc_margin = ArcFaceLoss(
            in_features=config.embedding_size,  # Changed from face_embedding_size
            out_features=2,
            scale=config.arcface_scale,
            margin=config.arcface_margin,
            easy_margin=config.easy_margin
        )
        
    def forward(self, anchor, other, labels=None):
        # Extract features
        anchor_features = self.feature_extractor(anchor)
        other_features = self.feature_extractor(other)
        
        # Use cosine similarity instead of absolute difference
        similarity = F.cosine_similarity(anchor_features, other_features, dim=1)
        similarity = similarity.view(-1, 1)  # Reshape to [batch_size, 1]
        
        # Concatenate features and reduce dimension
        embedding_combined = torch.cat([anchor_features, other_features], dim=1)
        embedding_combined = self.fusion_layer(embedding_combined)
        
        if labels is not None:
            # Training mode with ArcFace
            output = self.arc_margin(embedding_combined, labels)
        else:
            # Inference mode
            output = F.linear(F.normalize(embedding_combined), F.normalize(self.arc_margin.weight))
            output = output * self.config.arcface_scale
        
        return {
            'output': output,
            'anchor_features': anchor_features,
            'other_features': other_features,
            'similarity': similarity
        }

# Advanced training utilities (continued)
class TrainingUtils:
    @staticmethod
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr, cosine_decay)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, metrics, config, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': config.__dict__
        }, path)

    @staticmethod
    def load_checkpoint(path, model, optimizer=None, scheduler=None):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint

# Training functions
def train_epoch(model, train_loader, optimizer, scheduler, config, epoch, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        anchor = batch['anchor'].to(device, dtype=torch.float32)
        other = batch['other'].to(device, dtype=torch.float32)
        labels = batch['is_related'].to(device)
        
        try:
            if config.use_amp and scaler is not None:
                with autocast():
                    outputs = model(anchor, other, labels)
                    loss = criterion(outputs['output'], labels)
                    
                # Scale loss and check for NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    continue
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(anchor, other, labels)
                loss = criterion(outputs['output'], labels)
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs['output'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss / len(train_loader), acc

def validate(model, val_loader, config):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            labels = batch['is_related'].long().to(device)
            
            outputs = model(anchor, other)
            loss = criterion(outputs['output'], labels)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs['output'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and labels for metrics
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = 100. * correct / total
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    metrics = {
        'val_loss': total_loss / len(val_loader),
        'val_acc': acc,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1
    }
    
    return metrics
def create_dataloaders(config):
    # Create datasets
    train_dataset = KinshipDataset(config.train_path, config)
    val_dataset = KinshipDataset(config.val_path, config)
    test_dataset = KinshipDataset(config.test_path, config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
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
    
    return train_loader, val_loader, test_loader


# Training loop
def train_model(model, train_loader, val_loader, config):
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )
     # Warmup routine
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=len(train_loader) * config.warmup_epochs
    )
    # Calculate number of training steps
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = len(train_loader) * config.warmup_epochs
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.T_0,
        T_mult=config.T_mult,
        eta_min=config.min_lr
    )
    
    # Initialize mixed precision training
    scaler = GradScaler() if config.use_amp else None
    
    # Training history
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(config.num_epochs):
        
        curr_scheduler = warmup_scheduler if epoch < config.warmup_epochs else scheduler
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, curr_scheduler,
            config, epoch, scaler
        )

        
        # Step scheduler after each epoch
        if epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Validate
        val_metrics = validate(model, val_loader, config)
        
        # Update history and save checkpoints...
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_acc'].append(val_metrics['val_acc'])
        history['val_precision'].append(val_metrics['val_precision'])
        history['val_recall'].append(val_metrics['val_recall'])
        history['val_f1'].append(val_metrics['val_f1'])
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.2f}%")
        print(f"Val Precision: {val_metrics['val_precision']:.4f}")
        print(f"Val Recall: {val_metrics['val_recall']:.4f}")
        print(f"Val F1: {val_metrics['val_f1']:.4f}")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            TrainingUtils.save_checkpoint(
                model, optimizer, scheduler,
                epoch, val_metrics['val_loss'],
                val_metrics, config,
                os.path.join(model_dir, 'best_model.pth')
            )
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            TrainingUtils.save_checkpoint(
                model, optimizer, scheduler,
                epoch, val_metrics['val_loss'],
                val_metrics, config,
                os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
    
    return history

# Visualization utilities
def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Accuracy')
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    # Initialize configuration
    config = KinshipConfig()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    model = KinshipModel(config).to(device)
    
    # Train model
    history = train_model(model, train_loader, val_loader, config)
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    print("\nEvaluating final model...")
    final_metrics = validate(model, test_loader, config)
    print("\nFinal Test Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")