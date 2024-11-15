import os
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model_v6_1'
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

class KinshipConfig:
    def __init__(self):
        # Existing settings with updates
        self.input_size = 224
        self.face_embedding_size = 1024
        self.batch_size = 64
        self.learning_rate = 2e-4
        self.weight_decay = 1e-3
        self.num_epochs = 11
        self.warmup_epochs = 5
        
        # New settings
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2
        self.dropout_rate = 0.5

        
        # Paths remain the same
        self.train_path = '../data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
        self.val_path = '../data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
        self.test_path = '../data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'

def calculate_dataset_statistics(csv_path, config, num_samples=100000):
    """Calculate mean and std from a subset of the dataset"""
    data = pd.read_csv(csv_path)
    
    # Combine all image paths
    image_paths = []
    for _, row in data.iterrows():
        image_paths.extend([row['Anchor'], row['Positive'], row['Negative']])
    
    # Remove duplicates and shuffle
    image_paths = list(set(image_paths))
    np.random.shuffle(image_paths)
    
    # Take subset of images
    image_paths = image_paths[:num_samples]
    
    # Initialize arrays for mean and std calculation
    mean = np.zeros(3)
    std = np.zeros(3)
    
    # Basic transforms for consistent size
    basic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor()
    ])
    
    print("Calculating dataset statistics...")
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
    
    # Second pass: std
    for img_path in tqdm(image_paths, desc="Calculating std"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = basic_transform(img)
            std += torch.mean((img - torch.tensor(mean).view(3,1,1))**2, dim=(1,2)).numpy()
        except Exception as e:
            continue
    
    std = np.sqrt(std / n_images)
    
    return mean, std

class ImageProcessor:
    def __init__(self, config, is_training=False):
        self.config = config
        self.is_training = is_training
        
        # Calculate dataset statistics if not already computed
        stats_file = os.path.join(os.path.dirname(config.train_path), 'dataset_stats.npz')
        if os.path.exists(stats_file):
            stats = np.load(stats_file)
            self.mean = stats['mean']
            self.std = stats['std']
        else:
            print("Computing dataset statistics...")
            self.mean, self.std = calculate_dataset_statistics(config.train_path, config)
            # Save statistics for future use
            np.savez(stats_file, mean=self.mean, std=self.std)
        
        print(f"Dataset statistics - Mean: {self.mean}, Std: {self.std}")
        
        # Define transforms with computed statistics
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # transforms.RandomRotation(10),
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
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
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
        
        # Create balanced pairs from triplets
        pos_pairs = []
        neg_pairs = []
        
        for _, row in self.data.iterrows():
            pos_pairs.append((row['Anchor'], row['Positive'], 1))
            neg_pairs.append((row['Anchor'], row['Negative'], 0))
        
        # Ensure equal number of positive and negative pairs
        min_pairs = min(len(pos_pairs), len(neg_pairs))
        if is_training:  # Only balance training set
            pos_pairs = pos_pairs[:min_pairs]
            neg_pairs = neg_pairs[:min_pairs]
            
        self.pairs = pos_pairs + neg_pairs
        
        if is_training:
            random.shuffle(self.pairs)
        
        print(f"Loaded {len(self.pairs)} pairs")
        print("\nKinship distribution:")
        print(pd.Series([pair[2] for pair in self.pairs]).value_counts())
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_path, other_path, is_related = self.pairs[idx]
        
        # Process images
        anchor = self.processor.process_face(anchor_path)
        other = self.processor.process_face(other_path)
        
        if anchor is None or other is None:
            return self.__getitem__((idx + 1) % len(self))
        
        return {
            'anchor': anchor,
            'other': other,
            'is_related': torch.tensor(is_related, dtype=torch.float)
        }

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x1, x2):
        # Add batch dimension if necessary
        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)  # [B, 1, D]
        if x2.dim() == 2:
            x2 = x2.unsqueeze(1)  # [B, 1, D]
            
        # Compute Q, K, V
        q = self.query(x1)  # [B, 1, D]
        k = self.key(x2)    # [B, 1, D]
        v = self.value(x2)  # [B, 1, D]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, 1, 1]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = (attn @ v).squeeze(1)  # [B, D]
        return out

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
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

class KinshipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature Extractor
        self.feature_extractor = InceptionResnetV1(pretrained='vggface2')
        
        # Freeze early layers
        trainable_layers = 8
        for param in list(self.feature_extractor.parameters())[:-trainable_layers]:
            param.requires_grad = False
        
        # Adjust feature size if needed
        self.feature_proj = nn.Linear(512, config.face_embedding_size)
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            CrossAttention(config.face_embedding_size) 
            for _ in range(3)
        ])
        
        # Enhanced fusion network
        fusion_size = config.face_embedding_size * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, fusion_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            ResidualBlock(fusion_size, config.dropout_rate),
            nn.Linear(fusion_size, fusion_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            ResidualBlock(fusion_size // 2, config.dropout_rate)
        )
        
        # Kinship verification head
        self.kinship_verifier = nn.Sequential(
            nn.Linear(fusion_size // 2, fusion_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(fusion_size // 4, 1)
        )
    
    def forward(self, anchor, other):
        # Extract features
        anchor_features = self.feature_extractor(anchor)  # [B, 512]
        other_features = self.feature_extractor(other)    # [B, 512]
        
        # Project features to higher dimension if needed
        anchor_features = self.feature_proj(anchor_features)  # [B, 1024]
        other_features = self.feature_proj(other_features)    # [B, 1024]
        
        # L2 normalize embeddings
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        other_features = F.normalize(other_features, p=2, dim=1)
        
        # Apply cross-attention
        anchor_attended = anchor_features
        other_attended = other_features
        
        for attn in self.cross_attention:
            # Update features with attended versions
            anchor_update = attn(anchor_attended, other_attended)
            other_update = attn(other_attended, anchor_attended)
            
            # Residual connection
            anchor_attended = anchor_attended + anchor_update
            other_attended = other_attended + other_update
        
        # Concatenate features
        pair_features = torch.cat([anchor_attended, other_attended], dim=1)
        
        # Apply fusion network
        fused_features = self.fusion(pair_features)
        
        # Get kinship prediction
        kinship_score = self.kinship_verifier(fused_features)
        
        return {
            'kinship_score': kinship_score.squeeze(),
            'anchor_features': anchor_attended,
            'other_features': other_attended
        }
    
    

class KinshipLoss:
    def __init__(self, config):
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=0.3
        )
        self.label_smoothing = config.label_smoothing
        
    def _smooth_labels(self, targets):
        return targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
    def compute_loss(self, predictions, targets):
        # Apply label smoothing
        smoothed_targets = self._smooth_labels(targets['is_related'])
        
        # Main kinship loss
        kinship_loss = self.bce_loss(predictions['kinship_score'], smoothed_targets)
        
        # Feature similarity loss
        batch_size = predictions['anchor_features'].size(0)
        pos_mask = targets['is_related'] == 1
        neg_mask = targets['is_related'] == 0
        
        if torch.sum(pos_mask) > 0 and torch.sum(neg_mask) > 0:
            # Handle case where we might have different numbers of positives and negatives
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            
            # Ensure we have enough samples
            num_triplets = min(len(pos_indices), len(neg_indices), batch_size)
            
            # Randomly sample if we have more than we need
            if len(pos_indices) > num_triplets:
                pos_indices = pos_indices[torch.randperm(len(pos_indices))[:num_triplets]]
            if len(neg_indices) > num_triplets:
                neg_indices = neg_indices[torch.randperm(len(neg_indices))[:num_triplets]]
            
            # Get the features for triplet loss
            anchor = predictions['anchor_features'][:num_triplets]
            positive = predictions['other_features'][pos_indices[:num_triplets]]
            negative = predictions['other_features'][neg_indices[:num_triplets]]
            
            # Compute triplet loss
            triplet_loss = self.triplet_loss(anchor, positive, negative)
            
            total_loss = kinship_loss + 0.3 * triplet_loss
        else:
            total_loss = kinship_loss
        
        return total_loss

# You might also want to update the train_epoch function to handle optimizers correctly:
def train_epoch(model, train_loader, optimizers, loss_fn, device, scaler):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    feature_optimizer, main_optimizer = optimizers
    progress_bar = tqdm(train_loader, desc='Training')

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        anchor = batch['anchor'].to(device)
        other = batch['other'].to(device)
        is_related = batch['is_related'].to(device)
        
        # Use mixed precision training
        with autocast():
            predictions = model(anchor, other)
            loss = loss_fn.compute_loss(predictions, {'is_related': is_related})
        
        # Backward pass with gradient scaling
        feature_optimizer.zero_grad()
        main_optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        
        scaler.step(feature_optimizer)
        scaler.step(main_optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Compute batch accuracy
        pred_labels = (torch.sigmoid(predictions['kinship_score']) > 0.5).float()
        batch_correct = (pred_labels == is_related).sum().item()
        total_correct += batch_correct
        total_samples += is_related.size(0)

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        avg_accuracy = total_correct / total_samples
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}', 
            'acc': f'{avg_accuracy:.4f}'
        })

    return total_loss / len(train_loader), total_correct / total_samples
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature Extractor
        self.feature_extractor = InceptionResnetV1(pretrained='vggface2')
        
        # Freeze early layers
        trainable_layers = 8
        for param in list(self.feature_extractor.parameters())[:-trainable_layers]:
            param.requires_grad = False
        
        # Adjust feature size if needed
        self.feature_proj = nn.Linear(512, config.face_embedding_size)
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            CrossAttention(config.face_embedding_size) 
            for _ in range(3)
        ])
        
        # Enhanced fusion network
        fusion_size = config.face_embedding_size * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_size),
            nn.Linear(fusion_size, fusion_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            ResidualBlock(fusion_size, config.dropout_rate),
            nn.Linear(fusion_size, fusion_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            ResidualBlock(fusion_size // 2, config.dropout_rate)
        )
        
        # Kinship verification head
        self.kinship_verifier = nn.Sequential(
            nn.Linear(fusion_size // 2, fusion_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(fusion_size // 4, 1)
        )
    
    def forward(self, anchor, other):
        # Extract features
        anchor_features = self.feature_extractor(anchor)  # [B, 512]
        other_features = self.feature_extractor(other)    # [B, 512]
        
        # Project features to higher dimension if needed
        anchor_features = self.feature_proj(anchor_features)  # [B, 1024]
        other_features = self.feature_proj(other_features)    # [B, 1024]
        
        # L2 normalize embeddings
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        other_features = F.normalize(other_features, p=2, dim=1)
        
        # Apply cross-attention
        anchor_attended = anchor_features
        other_attended = other_features
        
        for attn in self.cross_attention:
            # Update features with attended versions
            anchor_update = attn(anchor_attended, other_attended)
            other_update = attn(other_attended, anchor_attended)
            
            # Residual connection
            anchor_attended = anchor_attended + anchor_update
            other_attended = other_attended + other_update
        
        # Concatenate features
        pair_features = torch.cat([anchor_attended, other_attended], dim=1)
        
        # Apply fusion network
        fused_features = self.fusion(pair_features)
        
        # Get kinship prediction
        kinship_score = self.kinship_verifier(fused_features)
        
        return {
            'kinship_score': kinship_score.squeeze(),
            'anchor_features': anchor_attended,
            'other_features': other_attended
        }


def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(val_loader, desc='Validation')

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            
            # Forward pass
            predictions = model(anchor, other)
            loss = loss_fn.compute_loss(predictions, {'is_related': is_related})
            
            # Calculate accuracy
            pred_labels = (torch.sigmoid(predictions['kinship_score']) > 0.5).float()
            total_correct += (pred_labels == is_related).sum().item()
            total_samples += is_related.size(0)
            
            total_loss += loss.item()

            # Compute batch accuracy
            pred_labels = (torch.sigmoid(predictions['kinship_score']) > 0.5).float()
            batch_correct = (pred_labels == is_related).sum().item()
            batch_accuracy = batch_correct / is_related.size(0)
            total_correct += batch_correct
            total_samples += is_related.size(0)

            # Compute average loss and accuracy so far
            avg_loss = total_loss / (batch_idx + 1)
            avg_accuracy = total_correct / total_samples

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_accuracy:.4f}'})

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# Modify the train_model function to include early stopping
def train_model(model, train_loader, val_loader, config, start_epoch=0, best_val_loss=float('inf')):
    # Separate optimizers for feature extractor and rest of network
    feature_optimizer = torch.optim.AdamW(
        model.feature_extractor.parameters(),
        lr=config.learning_rate * 0.1,
        weight_decay=config.weight_decay
    )
    
    main_optimizer = torch.optim.AdamW([
        {'params': model.cross_attention.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.kinship_verifier.parameters()}
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Schedulers
    feature_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        feature_optimizer, T_0=5, T_mult=2
    )
    
    main_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        main_optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs - start_epoch,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    loss_fn = KinshipLoss(config)
    scaler = GradScaler()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.0001)
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training
        model.train()
        train_loss, train_acc = train_epoch(
            model, train_loader, [feature_optimizer, main_optimizer], 
            loss_fn, device, scaler
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        
        # Update schedulers
        feature_scheduler.step()
        main_scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Learning Rate: {main_optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_optimizer_state_dict': feature_optimizer.state_dict(),
                'main_optimizer_state_dict': main_optimizer.state_dict(),
                'feature_scheduler_state_dict': feature_scheduler.state_dict(),
                'main_scheduler_state_dict': main_scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(model_dir, 'best_model.pth'))
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_optimizer_state_dict': feature_optimizer.state_dict(),
                'main_optimizer_state_dict': main_optimizer.state_dict(),
                'feature_scheduler_state_dict': feature_scheduler.state_dict(),
                'main_scheduler_state_dict': main_scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoints_dir, f'checkpoint_{epoch+1}.pth'))
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return best_val_loss

# Add create_dataloaders function
def create_dataloaders(config):
    print("Creating train dataset...")
    train_dataset = KinshipDataset(config.train_path, config, is_training=True)
    print("Creating validation dataset...")
    val_dataset = KinshipDataset(config.val_path, config, is_training=False)
    print("Creating test dataset...")
    test_dataset = KinshipDataset(config.test_path, config, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
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

# Add evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            
            # Forward pass
            predictions = model(anchor, other)
            pred_probs = torch.sigmoid(predictions['kinship_score'])
            pred_labels = (pred_probs > 0.5).float()
            
            # Calculate accuracy
            total_correct += (pred_labels == is_related).sum().item()
            total_samples += is_related.size(0)
            
            # Store predictions and labels for further metrics
            all_predictions.extend(pred_probs.cpu().numpy())
            all_labels.extend(is_related.cpu().numpy())
    
    # Calculate metrics
    accuracy = total_correct / total_samples
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate ROC AUC
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
    roc_auc = roc_auc_score(all_labels, all_predictions)
    
    # Calculate Average Precision (AP)
    ap = average_precision_score(all_labels, all_predictions)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)
    
    # Print results
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'average_precision': ap,
        'predictions': all_predictions,
        'labels': all_labels
    }

# Modify main script to include evaluation
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    config = KinshipConfig()
    train_loader, val_loader, test_loader = create_dataloaders(config)
    model = KinshipModel(config).to(device)
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming training from epoch {start_epoch}")
        
    
    # Train model
    train_model(model, train_loader, val_loader, config, 
                start_epoch=start_epoch, best_val_loss=best_val_loss)
    
    # Load best model for evaluation
    best_model = KinshipModel(config).to(device)
    best_model.load_state_dict(torch.load(os.path.join(model_dir, 'best_kin_nonkin_model.pth'))['model_state_dict'])
    
    # Evaluate on test set
    test_results = evaluate_model(best_model, test_loader, device)
    
    # Save test results
    np.savez(os.path.join(output_dir, 'test_results.npz'),
             predictions=test_results['predictions'],
             labels=test_results['labels'])
