#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# Verify device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model2'
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, 'model')
os.makedirs(model_dir, exist_ok=True)
checkpoints_dir = os.path.join(model_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

# In[2]:


class KinshipConfig:
    def __init__(self):
        # Model architecture
        self.input_size = 112  # Input image size
        self.face_embedding_size = 512
        
        # Training settings
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 5e-4
        self.num_epochs = 25
        
        # Data settings
        self.train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
        self.val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
        self.test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'


# In[3]:


# Custom image processing functions
class ImageProcessor:
    @staticmethod
    def read_image(path):
        """Read image using OpenCV and convert to RGB"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def resize_image(img, size):
        """Resize image keeping aspect ratio"""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h))
    
    @staticmethod
    def pad_image(img, size):
        """Pad image to square"""
        h, w = img.shape[:2]
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        return cv2.copyMakeBorder(
            img, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    
    @staticmethod
    def preprocess_image(img):
        """Normalize image to [-1, 1] range"""
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2
        return img
    
    @staticmethod
    def process_face(img_path, target_size=112):
        """Complete face processing pipeline"""
        try:
            # Read image
            img = ImageProcessor.read_image(img_path)
            
            # Resize keeping aspect ratio
            img = ImageProcessor.resize_image(img, target_size)
            
            # Pad to square
            img = ImageProcessor.pad_image(img, target_size)
            
            # Preprocess
            img = ImageProcessor.preprocess_image(img)
            
            # Convert to torch tensor
            img = torch.from_numpy(img.transpose(2, 0, 1))
            
            return img
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None


# In[4]:
# Dataset class
class KinshipDataset(Dataset):
    def __init__(self, csv_path, config):
        self.data = pd.read_csv(csv_path)
        self.config = config
        self.processor = ImageProcessor()
        
        # Create pairs from triplets
        self.pairs = []
        for _, row in self.data.iterrows():
            self.pairs.append((row['Anchor'], row['Positive'], 1))  # Kin pair
            self.pairs.append((row['Anchor'], row['Negative'], 0))  # Non-kin pair
        
        print(f"Loaded {len(self.pairs)} pairs")
        print("\nKinship distribution:")
        kin_counts = pd.Series([pair[2] for pair in self.pairs]).value_counts()
        print(kin_counts)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_path, other_path, is_related = self.pairs[idx]
        
        # Process images
        anchor = self.processor.process_face(anchor_path)
        other = self.processor.process_face(other_path)
        
        if anchor is None or other is None:
            # Return a default item if processing fails
            return self.__getitem__((idx + 1) % len(self))
        
        return {
            'anchor': anchor,
            'other': other,
            'is_related': torch.tensor(is_related, dtype=torch.float)
        }


# In[5]:
# Create data loaders
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


# In[6]:


# In[7]:
# Model components - Blocks
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
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
    """Residual Block with SE attention"""
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


# In[8]:

# Feature Extractor Network
class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, config.face_embedding_size)
        self.dropout = nn.Dropout(0.5)

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

    def forward(self, x):
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
        x = self.fc(x)
        
        return F.normalize(x, p=2, dim=1)


# In[9]:


# Kinship Verification Model
class KinshipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extractor (shared weights)
        self.feature_extractor = FeatureExtractor(config)
        
        # Fusion layers
        fusion_size = config.face_embedding_size * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, fusion_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_size // 2, fusion_size // 4),
            nn.ReLU(inplace=True)
        )
        
        # Kinship verification head
        hidden_size = fusion_size // 4
        self.kinship_verifier = nn.Linear(hidden_size, 1)

    def forward(self, anchor, other):
        # Extract features
        anchor_features = self.feature_extractor(anchor)
        other_features = self.feature_extractor(other)
        
        # Concatenate features
        pair_features = torch.cat([anchor_features, other_features], dim=1)
        
        # Fuse features
        fused_features = self.fusion(pair_features)
        
        # Get kinship score
        kinship_score = self.kinship_verifier(fused_features)
        
        return {
            'kinship_score': kinship_score.squeeze(),
            'anchor_features': anchor_features,
            'other_features': other_features
        }


# In[10]:


# Loss functions
class KinshipLoss:
    def __init__(self, config):
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_loss(self, predictions, targets):
        # Kinship verification loss
        kinship_loss = self.bce_loss(
            predictions['kinship_score'],
            targets['is_related']
        )
        
        return kinship_loss


# In[11]:


# Training functions
def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        # Move data to device
        anchor = batch['anchor'].to(device)
        other = batch['other'].to(device)
        is_related = batch['is_related'].to(device)
        
        # Forward pass
        predictions = model(anchor, other)
        
        # Compute loss
        loss = loss_fn.compute_loss(predictions, {'is_related': is_related})
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            
            # Forward pass
            predictions = model(anchor, other)
            
            # Compute loss
            loss = loss_fn.compute_loss(predictions, {'is_related': is_related})
            
            # Update metrics
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


# In[19]:


import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, config, start_epoch=0, best_val_loss=float('inf')):
    # Setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    loss_fn = KinshipLoss(config)
    train_loss_history = []
    val_loss_history = []
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_loss_history.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        val_loss_history.append(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
            }, os.path.join(model_dir, 'best_kin_nonkin_model.pth'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }, os.path.join(checkpoints_dir, f'checkpoint_{epoch}.pth'))




from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

def evaluate_model(model, test_loader, config):
    model.eval()
    predictions = []
    labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            
            # Get predictions
            outputs = model(anchor, other)
            kinship_score = outputs['kinship_score']
            
            # Convert logits to probabilities
            kinship_prob = torch.sigmoid(kinship_score)
            
            # Store results
            probabilities.extend(kinship_prob.cpu().numpy())
            predictions.extend((kinship_prob > 0.5).cpu().numpy())
            labels.extend(is_related.cpu().numpy())
    
    # Convert to numpy arrays
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    roc_auc = roc_auc_score(labels, probabilities)
    
    # Print results
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Evaluate based on sureness rate
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        sure_predictions = (probabilities > threshold).astype(int)
        sure_accuracy = accuracy_score(labels, sure_predictions)
        sure_precision, sure_recall, sure_f1, _ = precision_recall_fscore_support(labels, sure_predictions, average='binary')
        print(f"\nThreshold: {threshold}")
        print(f"Accuracy: {sure_accuracy:.4f}")
        print(f"Precision: {sure_precision:.4f}")
        print(f"Recall: {sure_recall:.4f}")
        print(f"F1 Score: {sure_f1:.4f}")
    
    return probabilities, labels


def load_model_and_plot_history(checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Create model
    config = KinshipConfig()
    model = KinshipModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if needed
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load history
    train_loss_history = checkpoint['train_loss_history']
    val_loss_history = checkpoint['val_loss_history']
    
    # Plot history
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))  # Save plot
    plt.show()
    
    return model

# Main training script
if __name__ == "__main__":
    # Initialize config
    config = KinshipConfig()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    model = KinshipModel(config).to(device)

    # Load checkpoint if exists
    checkpoint_path = os.path.join(model_dir, 'best_kin_nonkin_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
    
    # Train model
    train_model(model, train_loader, val_loader, config,start_epoch=start_epoch, best_val_loss=best_val_loss)
    
    print("Training completed!")

    model = load_model_and_plot_history(checkpoint_path)

    # Run evaluation

    # Load best model
    checkpoint = torch.load('/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model/model/best_kin_nonkin_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
        
    # Evaluate model
    probabilities, labels = evaluate_model(model, test_loader, config)


    





