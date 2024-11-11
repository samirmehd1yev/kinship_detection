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
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import logging
import wandb  # for experiment tracking
from torch.nn.modules.loss import _WeightedLoss
import random
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Directory setup
output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model_v6'
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

class KinshipConfig:
    def __init__(self):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_clip_val = 1.0
        self.checkpoint_frequency = 5  
        
        # Model architecture
        self.input_size = 160  # FaceNet expects 160x160
        self.face_embedding_size = 512  # Final embedding size
        self.initial_channels = 1792  # InceptionResnetV1's final channels
        
        # SE block parameters
        self.se_reduction = 16
        self.attention_heads = 8
        
        # Training settings
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.weight_decay = 2e-4
        self.num_epochs = 100
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        
        # Early stopping
        self.patience = 10
        self.min_delta = 1e-4
        
        # Loss weights
        self.triplet_weight = 0.4
        self.cosine_weight = 0.2
        self.center_weight = 0.1
        
        # Regularization
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        self.mixup_prob = 0.5
        self.gradient_clip_val = 1.0
        self.dropout_rate = 0.3
        
        # Data augmentation
        self.augment_prob = 0.5
        self.color_jitter_params = {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1
        }
        
        # Paths
        self.train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
        self.val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
        self.test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model_state):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model_state
            return False
        
        if self.mode == 'min':
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.best_state = model_state
            else:
                self.counter += 1
        else:
            if val_loss > self.best_loss + self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.best_state = model_state
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

class LabelSmoothingCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss/n, nll)

class MixupAugmentation:
    def __init__(self, config):
        self.alpha = config.mixup_alpha
    
    def mixup_data(self, x1, x2, y, device):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x1.size()[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x1 = lam * x1 + (1 - lam) * x1[index]
        mixed_x2 = lam * x2 + (1 - lam) * x2[index]
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x_reshaped = x.unsqueeze(1)  # Add sequence length dimension
        attn_output, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)
        return self.norm(attn_output.squeeze(1) + x)  # Add residual connection

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(in_channels // reduction_ratio, 8)  # Ensure minimum of 8 channels
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Global average pooling
        y = self.avg_pool(x).view(batch_size, self.in_channels)
        # Channel attention
        y = self.fc(y)
        # Reshape to match input dimensions
        y = y.view(batch_size, self.in_channels, 1, 1)
        # Scale the input
        return x * y.expand_as(x)

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        
        # Define channel sizes based on InceptionResnetV1 architecture
        self.channel_sizes = {
            'block35': 256,
            'block17': 896,
            'block8': 1792,
            'final': 1792
        }
        
        # SE blocks for each feature level
        self.se_blocks = nn.ModuleDict({
            'block35': SEBlock(self.channel_sizes['block35']),
            'block17': SEBlock(self.channel_sizes['block17']),
            'block8': SEBlock(self.channel_sizes['block8']),
            'final': SEBlock(self.channel_sizes['final'])
        })
        
        # Final processing to get embedding
        self.final_processing = nn.Sequential(
            nn.Conv2d(self.channel_sizes['final'], config.face_embedding_size, 1),
            nn.BatchNorm2d(config.face_embedding_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Freeze early layers
        self._freeze_layers(8)  # Keep last 8 layers trainable
    
    def _freeze_layers(self, trainable_layers):
        """Freeze all layers except the last trainable_layers"""
        layers_to_freeze = list(self.facenet.parameters())[:-trainable_layers]
        for param in layers_to_freeze:
            param.requires_grad = False
    
    def forward(self, x):
        # Initial processing
        x = self.facenet.conv2d_1a(x)
        x = self.facenet.conv2d_2a(x)
        x = self.facenet.conv2d_2b(x)
        x = self.facenet.maxpool_3a(x)
        x = self.facenet.conv2d_3b(x)
        x = self.facenet.conv2d_4a(x)
        x = self.facenet.conv2d_4b(x)
        
        # Block35 processing
        identity = x
        x = self.facenet.repeat_1(x)
        x = self.se_blocks['block35'](x)
        x = x + identity if x.size() == identity.size() else x
        
        # Block17 processing
        x = self.facenet.mixed_6a(x)
        identity = x
        x = self.facenet.repeat_2(x)
        x = self.se_blocks['block17'](x)
        x = x + identity if x.size() == identity.size() else x
        
        # Block8 processing
        x = self.facenet.mixed_7a(x)
        identity = x
        x = self.facenet.repeat_3(x)
        x = self.se_blocks['block8'](x)
        x = x + identity if x.size() == identity.size() else x
        
        # Final processing
        x = self.facenet.block8(x)
        x = self.se_blocks['final'](x)
        
        # Get final embedding
        x = self.final_processing(x)
        x = x.view(x.size(0), -1)
        
        return x


class EnhancedFusionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        fusion_size = config.face_embedding_size * 2
        
        self.initial_norm = nn.BatchNorm1d(fusion_size)
        
        # Multi-scale feature processing
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_size, fusion_size // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate)
            ),
            nn.Sequential(
                nn.Linear(fusion_size, fusion_size // 4),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate)
            ),
            nn.Sequential(
                nn.Linear(fusion_size, fusion_size // 8),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate)
            )
        ])
        
        # Cross attention
        self.cross_attention = MultiHeadAttention(fusion_size, num_heads=8)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_size * 7 // 8, fusion_size // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout_rate)
        )
    
    def forward(self, x):
        x = self.initial_norm(x)
        branch_outputs = [branch(x) for branch in self.branches]
        attended = self.cross_attention(x)
        combined = torch.cat([*branch_outputs, attended], dim=1)
        return self.output_proj(combined)

class ImprovedKinshipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extractor
        self.feature_extractor = EnhancedFeatureExtractor(config)
        
        # Fusion network
        fusion_size = config.face_embedding_size * 2
        self.fusion = EnhancedFusionNetwork(config)
        
        # Contrastive head
        self.contrastive_head = nn.Sequential(
            nn.Linear(fusion_size // 4, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        # Verification head
        self.kinship_verifier = nn.Sequential(
            nn.BatchNorm1d(fusion_size // 4),
            nn.Linear(fusion_size // 4, fusion_size // 8),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout_rate),
            nn.LayerNorm(fusion_size // 8),
            nn.Linear(fusion_size // 8, fusion_size // 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout_rate / 2),
            nn.Linear(fusion_size // 16, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, anchor, other):
        # Extract features
        anchor_features = self.feature_extractor(anchor)
        other_features = self.feature_extractor(other)
        
        # L2 normalize embeddings
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        other_features = F.normalize(other_features, p=2, dim=1)
        
        # Concatenate features
        pair_features = torch.cat([anchor_features, other_features], dim=1)
        
        # Apply fusion network
        fused_features = self.fusion(pair_features)
        
        # Get contrastive embeddings
        contrastive_embeddings = self.contrastive_head(fused_features)
        
        # Get kinship score
        kinship_score = self.kinship_verifier(fused_features)
        
        return {
            'kinship_score': kinship_score.squeeze(),
            'anchor_features': anchor_features,
            'other_features': other_features,
            'contrastive_embeddings': contrastive_embeddings
        }

# Continue EnhancedKinshipLoss
class EnhancedKinshipLoss:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(config.device))
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3)
        self.contrastive_temp = 0.07
        self.lambda_triplet = 0.3
        self.lambda_contrastive = 0.1

    def contrastive_loss(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.t()) / self.contrastive_temp
        
        # Create labels matrix
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(embeddings.device)
        
        # Compute loss
        exp_logits = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Handle zero division
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        
        return loss

    def compute_loss(self, predictions, targets):
        # Ensure all tensors are on the same device
        device = predictions['kinship_score'].device
        targets = {k: v.to(device) if torch.is_tensor(v) else v 
                  for k, v in targets.items()}

        # Binary classification loss
        kinship_loss = self.bce_loss(
            predictions['kinship_score'],
            targets['is_related']
        )

        total_loss = kinship_loss
        loss_components = {'kinship_loss': kinship_loss.item()}

        # Contrastive loss if embeddings are available
        if 'contrastive_embeddings' in predictions:
            contrastive_loss = self.contrastive_loss(
                predictions['contrastive_embeddings'],
                targets['is_related']
            )
            total_loss += self.lambda_contrastive * contrastive_loss
            loss_components['contrastive_loss'] = contrastive_loss.item()

        # Feature similarity loss using triplet loss
        if all(k in predictions for k in ['anchor_features', 'other_features']):
            pos_mask = targets['is_related'] == 1
            neg_mask = targets['is_related'] == 0
            
            if torch.sum(pos_mask) > 0 and torch.sum(neg_mask) > 0:
                anchor_features = predictions['anchor_features']
                all_features = predictions['other_features']
                
                # Compute pairwise distances
                dist_matrix = torch.cdist(anchor_features, all_features, p=2)
                
                # Handle positive pairs
                pos_dist = dist_matrix * pos_mask.float()
                pos_dist = torch.where(pos_mask.float().bool(), pos_dist, 
                                     torch.full_like(pos_dist, float('inf')))
                
                # Handle negative pairs
                neg_dist = dist_matrix * neg_mask.float()
                neg_dist = torch.where(neg_mask.float().bool(), neg_dist,
                                     torch.full_like(neg_dist, float('inf')))
                
                # Find hardest positives and negatives
                hardest_pos_idx = torch.argmin(pos_dist, dim=1)
                hardest_neg_idx = torch.argmin(neg_dist, dim=1)
                
                # Get features for triplet loss
                positive_features = all_features[hardest_pos_idx]
                negative_features = all_features[hardest_neg_idx]
                
                triplet_loss = self.triplet_loss(
                    anchor_features,
                    positive_features,
                    negative_features
                )
                
                total_loss += self.lambda_triplet * triplet_loss
                loss_components['triplet_loss'] = triplet_loss.item()

        return total_loss, loss_components
    
class ImprovedDataset(Dataset):
    def __init__(self, csv_path, config, is_training=False):
        self.data = pd.read_csv(csv_path)
        self.config = config
        self.is_training = is_training
        
        # Enhanced transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(**config.color_jitter_params)
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ], p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
        ]) if is_training else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create balanced pairs
        self.pairs = []
        for _, row in self.data.iterrows():
            self.pairs.append((row['Anchor'], row['Positive'], 1))
            self.pairs.append((row['Anchor'], row['Negative'], 0))
            
        logger.info(f"Loaded {len(self.pairs)} pairs")
        logger.info("\nKinship distribution:")
        logger.info(pd.Series([pair[2] for pair in self.pairs]).value_counts())
        
    def __len__(self):
        return len(self.pairs)
    
    def _process_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(img)
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")
            return None
    
    def __getitem__(self, idx):
        anchor_path, other_path, is_related = self.pairs[idx]
        
        anchor = self._process_image(anchor_path)
        other = self._process_image(other_path)
        
        if anchor is None or other is None:
            return self.__getitem__((idx + 1) % len(self))
            
        return {
            'anchor': anchor,
            'other': other,
            'is_related': torch.tensor(is_related, dtype=torch.float)
        }

def train_one_epoch(model, train_loader, optimizer, criterion, scheduler, config, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    num_samples = 0
    loss_components = defaultdict(float)
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        batch = {k: v.to(config.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch['anchor'], batch['other'])
        
        # Compute loss
        loss, batch_loss_components = criterion.compute_loss(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if hasattr(config, 'gradient_clip_val'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        pred_labels = (torch.sigmoid(outputs['kinship_score']) > 0.5).float()
        accuracy = (pred_labels == batch['is_related']).float().mean().item()
        
        # Update metrics
        batch_size = batch['is_related'].size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy * batch_size
        num_samples += batch_size
        
        for k, v in batch_loss_components.items():
            loss_components[k] += v * batch_size
            
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })
        
        # Log to wandb
        wandb.log({
            'train/batch_loss': loss.item(),
            'train/batch_accuracy': accuracy,
            'train/learning_rate': optimizer.param_groups[0]['lr'],
            **{f'train/batch_{k}': v for k, v in batch_loss_components.items()}
        }, commit=True)
    
    # Compute epoch averages
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    avg_components = {k: v / num_samples for k, v in loss_components.items()}
    
    # Log epoch metrics
    wandb.log({
        'train/epoch_loss': avg_loss,
        'train/epoch_accuracy': avg_acc,
        **{f'train/epoch_{k}': v for k, v in avg_components.items()},
        'epoch': epoch
    }, commit=True)
    
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, config, epoch):
    model.eval()
    total_loss = 0
    total_acc = 0
    num_samples = 0
    loss_components = defaultdict(float)
    
    progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            batch = {k: v.to(config.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch['anchor'], batch['other'])
            
            # Compute loss
            loss, batch_loss_components = criterion.compute_loss(outputs, batch)
            
            # Calculate accuracy
            pred_labels = (torch.sigmoid(outputs['kinship_score']) > 0.5).float()
            accuracy = (pred_labels == batch['is_related']).float().mean().item()
            
            # Update metrics
            batch_size = batch['is_related'].size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy * batch_size
            num_samples += batch_size
            
            for k, v in batch_loss_components.items():
                loss_components[k] += v * batch_size
                
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    # Compute averages
    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    avg_components = {k: v / num_samples for k, v in loss_components.items()}
    
    # Log validation metrics
    wandb.log({
        'val/epoch_loss': avg_loss,
        'val/epoch_accuracy': avg_acc,
        **{f'val/epoch_{k}': v for k, v in avg_components.items()},
        'epoch': epoch
    }, commit=True)
    
    return avg_loss, avg_acc



def create_loaders(config):
    train_dataset = ImprovedDataset(config.train_path, config, is_training=True)
    val_dataset = ImprovedDataset(config.val_path, config, is_training=False)
    test_dataset = ImprovedDataset(config.test_path, config, is_training=False)
    
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
    
    return train_loader, val_loader, test_loader

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    config = KinshipConfig()
    
    # Initialize wandb
    wandb.init(project="kinship-verification", config={
        "architecture": "improved_kinship_model",
        "dataset": "FIW",
        "epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay
    })
    
    train_loader, val_loader, test_loader = create_loaders(config)
    
    # Create model and move to device
    model = ImprovedKinshipModel(config).to(config.device)
    
    # Setup training components
    criterion = EnhancedKinshipLoss(config)
    
    # Different learning rates for different components
    param_groups = [
        {'params': model.feature_extractor.parameters(), 'lr': config.learning_rate * 0.1},
        {'params': model.fusion.parameters()},
        {'params': model.contrastive_head.parameters()},
        {'params': model.kinship_verifier.parameters()}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[config.learning_rate * 0.1, config.learning_rate, 
                config.learning_rate, config.learning_rate],
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode='max'  # We want to maximize validation accuracy
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            config=config,
            epoch=epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
            epoch=epoch
        )
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': config,
            }, os.path.join(model_dir, 'best_model.pth'))
            
            logger.info(f"New best model saved! Validation Accuracy: {val_acc:.4f}")
        
        # Early stopping check
        if early_stopping(val_acc, model.state_dict()):
            logger.info("Early stopping triggered!")
            break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc = validate(
        model=model,
        val_loader=test_loader,  # Use test loader for final evaluation
        criterion=criterion,
        config=config,
        epoch=-1  # Use -1 to indicate final evaluation
    )
    
    logger.info(f"\nFinal Test Results:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Log final test metrics
    wandb.log({
        "test/final_loss": test_loss,
        "test/final_accuracy": test_acc
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()