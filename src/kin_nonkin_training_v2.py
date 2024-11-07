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
from tqdm import tqdm 
from torchvision import transforms
import math
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, roc_curve
from sklearn.manifold import TSNE  


# Verify device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Directory setup
output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model_v2'
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

class EnhancedKinshipConfig:
    """Enhanced configuration for the kinship model"""
    def __init__(self):
        # Model architecture
        self.input_size = 112
        self.face_embedding_size = 512
        self.num_classes = 1538  # Number of unique identities
        
        # Family-aware parameters
        self.family_margin = 0.3
        self.family_embedding_size = 256
        self.num_families = 378
        
        # Dataset statistics for FIW
        self.total_identities = 2231
        self.train_identities = 1537
        self.val_identities = 336
        self.test_identities = 358
        
        # ArcFace parameters
        self.arcface_scale = 30.0  # Reduced from 64.0
        self.arcface_margin = 0.3  # Reduced from 0.5
        
        # Loss parameters
        self.margin = 0.3
        self.triplet_margin = 0.3
        
        # Training settings
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.num_epochs = 10
        self.gradient_accumulation_steps = 4  # New parameter
        self.warmup_epochs = 5  # New parameter
        
        # Data augmentation settings
        self.use_augmentation = True
        self.random_horizontal_flip = 0.5
        self.random_brightness = 0.2
        self.random_contrast = 0.2
                
        # Data paths
        self.train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
        self.val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
        self.test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'

class EnhancedImageProcessor:
    """Enhanced image processing with augmentation"""
    def __init__(self, config):
        self.config = config
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=config.random_horizontal_flip),
            transforms.ColorJitter(
                brightness=config.random_brightness,
                contrast=config.random_contrast
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]) if config.use_augmentation else None
        
        self.test_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def read_image(path):
        """Read image using OpenCV and convert to RGB"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def preprocess_image(self, img, is_training=False):
        """Preprocess image with normalization and optional augmentation"""
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        if is_training and self.train_transform:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
            
        return img
    
    def process_face(self, img_path, is_training=False):
        """Complete face processing pipeline"""
        try:
            img = self.read_image(img_path)
            img = cv2.resize(img, (self.config.input_size, self.config.input_size))
            img = self.preprocess_image(img, is_training)
            return img
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None

class EnhancedKinshipDataset(Dataset):
    def __init__(self, csv_path, config, is_training=False):
        self.data = pd.read_csv(csv_path)
        self.config = config
        self.is_training = is_training
        self.processor = EnhancedImageProcessor(config)
        
        # Create pairs and extract identity information
        self.pairs = []
        self.identities = {}
        self.families = {}
        identity_counter = 0
        
        # Add a special "unknown" identity for negative pairs
        self.unknown_identity = identity_counter
        identity_counter += 1
        
        # First pass: Collect all unique identities and families
        print("First pass: Collecting unique identities...")
        for _, row in self.data.iterrows():
            # Process Anchor
            anchor_family = os.path.basename(os.path.dirname(os.path.dirname(row['Anchor'])))
            anchor_mid = os.path.basename(os.path.dirname(row['Anchor']))
            anchor_id = f"{anchor_family}/{anchor_mid}"
            
            # Process Positive
            positive_family = os.path.basename(os.path.dirname(os.path.dirname(row['Positive'])))
            positive_mid = os.path.basename(os.path.dirname(row['Positive']))
            positive_id = f"{positive_family}/{positive_mid}"
            
            # Add identities to mapping
            if anchor_id not in self.identities:
                self.identities[anchor_id] = identity_counter
                identity_counter += 1
            
            if positive_id not in self.identities:
                self.identities[positive_id] = identity_counter
                identity_counter += 1
            
            # Store family information
            if anchor_family not in self.families:
                self.families[anchor_family] = len(self.families)
            if positive_family not in self.families:
                self.families[positive_family] = len(self.families)
        
        # Second pass: Create pairs
        print("Second pass: Creating pairs...")
        for _, row in self.data.iterrows():
            # Process anchor
            anchor_family = os.path.basename(os.path.dirname(os.path.dirname(row['Anchor'])))
            anchor_mid = os.path.basename(os.path.dirname(row['Anchor']))
            anchor_id = f"{anchor_family}/{anchor_mid}"
            
            # Process positive
            positive_family = os.path.basename(os.path.dirname(os.path.dirname(row['Positive'])))
            positive_mid = os.path.basename(os.path.dirname(row['Positive']))
            positive_id = f"{positive_family}/{positive_mid}"
            
            # Create positive pair
            self.pairs.append({
                'anchor': row['Anchor'],
                'other': row['Positive'],
                'is_related': 1,
                'identity': self.identities[anchor_id],
                'other_identity': self.identities[positive_id],
                'family': self.families[anchor_family],
                'relationship': row['ptype'] if 'ptype' in row else None
            })
            
            # Create negative pair
            negative = row['Negative'].strip('*') if '*' in row['Negative'] else row['Negative']
            negative_family = os.path.basename(os.path.dirname(os.path.dirname(negative)))
            self.pairs.append({
                'anchor': row['Anchor'],
                'other': negative,
                'is_related': 0,
                'identity': self.identities[anchor_id],
                'other_identity': self.unknown_identity,
                'family': self.families[anchor_family],
                'relationship': 'unrelated'
            })
        
        print(f"Loaded {len(self.pairs)} pairs")
        print(f"Number of unique identities: {len(self.identities) + 1}")  # +1 for unknown_identity
        print(f"Number of families: {len(self.families)}")
        
        # Save identity mapping for debugging
        mapping_file = os.path.join(os.path.dirname(csv_path), 'identity_mapping.txt')
        with open(mapping_file, 'w') as f:
            f.write("ID Mapping:\n")
            for identity, idx in sorted(self.identities.items()):
                f.write(f"{identity}: {idx}\n")
            f.write(f"\nUnknown Identity: {self.unknown_identity}")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Process images
        anchor = self.processor.process_face(pair['anchor'], self.is_training)
        other = self.processor.process_face(pair['other'], self.is_training)
        
        if anchor is None or other is None:
            # Return a default item if processing fails
            return self.__getitem__((idx + 1) % len(self))
        
        return {
            'anchor': anchor,
            'other': other,
            'is_related': torch.tensor(pair['is_related'], dtype=torch.float),
            'identity': torch.tensor(pair['identity'], dtype=torch.long),
            'other_identity': torch.tensor(pair['other_identity'], dtype=torch.long),
            'family': torch.tensor(pair['family'], dtype=torch.long),
            'relationship': pair['relationship']
        }
# ArcFace and Attention Modules
class ArcMarginProduct(nn.Module):
    """ArcFace head for face recognition"""
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output, cosine

class MultiHeadAttention(nn.Module):
    """Multi-head Attention module for feature refinement"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size, embed_dim = x.size(0), x.size(1)
        x = x.view(1, batch_size, embed_dim)  # reshape for attention
        attn_output, _ = self.mha(x, x, x)
        attn_output = attn_output.view(batch_size, embed_dim)  # reshape back
        return self.norm(x.view(batch_size, embed_dim) + attn_output)

# Base Backbone Network (ResNet-based)
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Enhanced Residual Block with SE attention"""
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class EnhancedFeatureExtractor(nn.Module):
    """Enhanced Feature Extractor with ArcFace backbone"""
    def __init__(self, config):
        super().__init__()
        
        # Initial convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Attention and final layers
        self.attention = MultiHeadAttention(config.face_embedding_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, config.face_embedding_size)
        self.dropout = nn.Dropout(0.5)
        
        # ArcFace head
        self.arcface = ArcMarginProduct(
            config.face_embedding_size,
            config.num_classes,
            scale=config.arcface_scale,
            margin=config.arcface_margin
        )

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

        layers = []
        layers.append(ResidualBlock(in_c, out_c, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        # Backbone feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        features = self.fc(x)
        
        # Apply attention and normalize
        features = self.attention(features)
        features = F.normalize(features, p=2, dim=1)
        
        if labels is not None:
            # Get ArcFace outputs if labels are provided
            arcface_output, cosine = self.arcface(features, labels)
            return {
                'features': features,
                'arcface_output': arcface_output,
                'cosine': cosine
            }
        
        return features

class EnhancedKinshipModel(nn.Module):
    """Enhanced Kinship Verification Model with multiple losses"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced feature extractor with ArcFace
        self.feature_extractor = EnhancedFeatureExtractor(config)
        
        # Fusion network with residual connections
        fusion_size = config.face_embedding_size * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, fusion_size // 2),
            nn.BatchNorm1d(fusion_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_size // 2, fusion_size // 4),
            nn.BatchNorm1d(fusion_size // 4),
            nn.ReLU(inplace=True)
        )
        
        # Kinship verification head
        self.kinship_verifier = nn.Sequential(
            nn.Linear(fusion_size // 4, fusion_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_size // 8, 1)
        )

    def forward(self, anchor, other, labels=None):
        if labels is not None:
            # Get features with ArcFace outputs
            anchor_out = self.feature_extractor(anchor, labels)
            other_out = self.feature_extractor(other, labels)
            
            anchor_features = anchor_out['features']
            other_features = other_out['features']
            
            arcface_outputs = {
                'anchor_arcface': anchor_out['arcface_output'],
                'other_arcface': other_out['arcface_output'],
                'anchor_cosine': anchor_out['cosine'],
                'other_cosine': other_out['cosine']
            }
        else:
            # Get features only
            anchor_features = self.feature_extractor(anchor)
            other_features = self.feature_extractor(other)
            arcface_outputs = None
        
        # Concatenate and fuse features
        pair_features = torch.cat([anchor_features, other_features], dim=1)
        fused_features = self.fusion(pair_features)
        
        # Get kinship score
        kinship_score = self.kinship_verifier(fused_features)
        
        output = {
            'kinship_score': kinship_score.squeeze(),
            'anchor_features': anchor_features,
            'other_features': other_features,
            'fused_features': fused_features
        }
        
        if arcface_outputs:
            output.update(arcface_outputs)
            
        return output

class EnhancedKinshipLoss:
    def __init__(self, config):
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=config.triplet_margin)
        
        # Adjusted loss weights
        self.w_kinship = 1.0
        self.w_arcface = 0.2  # Reduced from 1.0
        self.w_triplet = 0.3  # Reduced from 0.5
        self.w_contrastive = 0.3
        self.w_family = 0.2  # Reduced from 0.5
        
    def contrastive_loss(self, anchor_features, other_features, is_related):
        """Compute contrastive loss"""
        distance = F.pairwise_distance(anchor_features, other_features)
        return torch.mean((1 - is_related) * torch.pow(distance, 2) + 
                         is_related * torch.pow(torch.clamp(self.config.margin - distance, min=0.0), 2))
        
    def family_loss(self, features, family_labels):
        """Compute loss based on family relationships"""
        pairwise_dist = torch.cdist(features, features)
        mask_same_family = (family_labels.unsqueeze(0) == family_labels.unsqueeze(1)).float()
        
        # Encourage features from same family to be closer
        family_loss = (mask_same_family * pairwise_dist).mean() + \
                     ((1 - mask_same_family) * torch.relu(self.config.margin - pairwise_dist)).mean()
        return family_loss
    
    def compute_loss(self, predictions, targets):
        total_loss = 0
        loss_components = {}
        
        # Kinship verification loss
        kinship_loss = self.bce_loss(
            predictions['kinship_score'],
            targets['is_related']
        )
        total_loss += self.w_kinship * kinship_loss
        loss_components['kinship_loss'] = kinship_loss.item()
        
        # Add ArcFace loss only for related pairs (is_related == 1)
        if 'anchor_arcface' in predictions:
            # Create mask for related pairs
            related_mask = targets['is_related'] == 1
            
            if related_mask.sum() > 0:  # Only compute if there are related pairs
                arcface_loss = (
                    self.ce_loss(
                        predictions['anchor_arcface'][related_mask],
                        targets['identity'][related_mask]
                    ) +
                    self.ce_loss(
                        predictions['other_arcface'][related_mask],
                        targets['other_identity'][related_mask]
                    )
                ) / 2
                total_loss += self.w_arcface * arcface_loss
                loss_components['arcface_loss'] = arcface_loss.item()
            else:
                loss_components['arcface_loss'] = 0.0
        
        # Rest of the loss computations remain the same
        contrastive_loss = self.contrastive_loss(
            predictions['anchor_features'],
            predictions['other_features'],
            targets['is_related']
        )
        total_loss += self.w_contrastive * contrastive_loss
        loss_components['contrastive_loss'] = contrastive_loss.item()
        
        # Family loss
        family_loss = self.family_loss(predictions['fused_features'], targets['family'])
        total_loss += self.w_family * family_loss
        loss_components['family_loss'] = family_loss.item()
        
        # Triplet loss (only for positive pairs)
        if targets['is_related'].sum() > 0:
            triplet_loss = online_triplet_mining(
                predictions['fused_features'],
                targets['identity'],
                margin=self.config.triplet_margin
            )
            total_loss += self.w_triplet * triplet_loss
            loss_components['triplet_loss'] = triplet_loss.item()
        
        return total_loss, loss_components

def create_dataloaders(config):
    """Create train, validation and test dataloaders"""
    train_dataset = EnhancedKinshipDataset(config.train_path, config, is_training=True)
    val_dataset = EnhancedKinshipDataset(config.val_path, config, is_training=False)
    test_dataset = EnhancedKinshipDataset(config.test_path, config, is_training=False)
    
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

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
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
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
def train_model_enhanced(model, train_loader, val_loader, config, optimizer, scheduler, start_epoch=0, best_val_loss=float('inf')):
    loss_fn = EnhancedKinshipLoss(config)
    early_stopping = EarlyStopping(patience=7)
    
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = len(train_loader) * config.warmup_epochs
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_loss = 0
        all_loss_components = {
            'kinship_loss': 0,
            'arcface_loss': 0,
            'contrastive_loss': 0,
            'triplet_loss': 0,
            'family_loss': 0
        }
        
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for i, batch in enumerate(progress_bar):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            identity = batch['identity'].to(device)
            family = batch['family'].to(device)
            other_identity = batch['other_identity'].to(device)
            
            # Forward pass
            predictions = model(anchor, other, identity)
            
            # Compute loss
            loss, loss_components = loss_fn.compute_loss(
                predictions,
                {
                    'is_related': is_related,
                    'identity': identity,
                    'other_identity': other_identity,
                    'family': family 
                }
            )
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights if needed
            if (i + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * config.gradient_accumulation_steps
            for k, v in loss_components.items():
                all_loss_components[k] += v
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss/(i+1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Validation phase
        val_loss = validate_enhanced(model, val_loader, loss_fn, device)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Learning rate scheduling
        if epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{config.num_epochs} Summary:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print('Loss Components:')
        for k, v in all_loss_components.items():
            print(f'  {k}: {v/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(model_dir, 'best_model.pth'))
            
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoints_dir, f'checkpoint_{epoch}.pth'))
                   
def validate_enhanced(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            identity = batch['identity'].to(device)
            family = batch['family'].to(device)  # Add family
            other_identity = batch['other_identity'].to(device)  # Add other identity
            
            predictions = model(anchor, other, identity)
            
            loss, _ = loss_fn.compute_loss(
                predictions,
                {
                    'is_related': is_related,
                    'identity': identity,
                    'other_identity': other_identity,
                    'family': family  # Add family to targets
                }
            )
            
            
            # Store predictions and labels for metrics
            kinship_probs = torch.sigmoid(predictions['kinship_score'])
            all_predictions.extend(kinship_probs.cpu().numpy())
            all_labels.extend(is_related.cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate additional metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, (all_predictions > 0.5).astype(int))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        (all_predictions > 0.5).astype(int), 
        average='binary'
    )
    auc_roc = roc_auc_score(all_labels, all_predictions)
    
    print(f"\nValidation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    return total_loss / len(val_loader)

def online_triplet_mining(features, labels, margin=0.3):
    """Online triplet mining with hard negative mining"""
    pairwise_dist = torch.cdist(features, features)
    
    # Create mask for valid positive and negative pairs
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.t()).float()
    mask_neg = (labels != labels.t()).float()
    
    # Remove self-pairs from positive mask
    mask_pos = mask_pos.fill_diagonal_(0)
    
    # Get hardest positive and negative pairs
    hardest_pos = torch.max(pairwise_dist * mask_pos, dim=1)[0]
    hardest_neg = torch.min(pairwise_dist * mask_neg + 1e5 * (1 - mask_neg), dim=1)[0]
    
    # Calculate triplet loss
    triplet_loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)
    
    # Only consider valid triplets (those with both positive and negative pairs)
    valid_mask = (hardest_pos > 0) & (hardest_neg > 0)
    if valid_mask.sum() > 0:
        triplet_loss = triplet_loss[valid_mask].mean()
    else:
        triplet_loss = torch.tensor(0.0).to(features.device)
    
    return triplet_loss

def evaluate_model_enhanced(model, test_loader, config):
    """Enhanced evaluation function with detailed metrics"""
    model.eval()
    predictions = []
    labels = []
    embeddings = []
    identities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            identity = batch['identity'].to(device)
            
            # Get predictions
            outputs = model(anchor, other)
            kinship_score = outputs['kinship_score']
            
            # Store features for analysis
            embeddings.extend(outputs['fused_features'].cpu().numpy())
            identities.extend(identity.cpu().numpy())
            
            # Convert logits to probabilities
            kinship_prob = torch.sigmoid(kinship_score)
            
            # Store results
            predictions.extend(kinship_prob.cpu().numpy())
            labels.extend(is_related.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    identities = np.array(identities)
    
    # Calculate metrics at different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        pred_labels = (predictions > threshold).astype(int)
        
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='binary'
        )
        
        print(f"\nMetrics at threshold {threshold}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(labels, predictions)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Return all metrics for further analysis
    return {
        'predictions': predictions,
        'labels': labels,
        'embeddings': embeddings,
        'identities': identities,
        'roc_auc': roc_auc
    }

def visualize_results(results, output_dir):
    """Visualize evaluation results"""
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(results['labels'], results['predictions'])
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {results["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot embedding visualization using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(results['embeddings'])
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=results['identities'], cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Face Embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.savefig(os.path.join(output_dir, 'embedding_visualization.png'))
    plt.close()

def main():
    # Initialize config
    config = EnhancedKinshipConfig()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    model = EnhancedKinshipModel(config).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize scheduler with warmup
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = len(train_loader) * config.warmup_epochs
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.num_epochs,
        pct_start=config.warmup_epochs/config.num_epochs
    )
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
    
    # Train model
    train_model_enhanced(model, train_loader, val_loader, config, optimizer, scheduler, start_epoch, best_val_loss)
    
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    results = evaluate_model_enhanced(model, test_loader, config)
    
    # Visualize results
    visualize_results(results, output_dir)
    
    print("\nTraining and evaluation completed!")
    print(f"Results and visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()