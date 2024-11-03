import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
        self.train_path = '../data/processed/fiw/train/splits/train_triplets.csv'
        self.val_path = '../data/processed/fiw/train/splits/val_triplets.csv'
        self.test_path = '../data/processed/fiw/train/splits/test_triplets.csv'

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
    def align_face(img):
        """Align face using InsightFace"""
        faces = app.get(img)
        if not faces:
            raise ValueError("No faces detected.")
        face = faces[0]
        kps = face.kps.astype(int)
        aligned_face = face_align.norm_crop(img, kps)
        if aligned_face is None:
            raise ValueError("Failed to align face.")
        return aligned_face
    
    @staticmethod
    def preprocess_image(img):
        """Normalize image to [-1, 1] range"""
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2
        return img
    
    @staticmethod
    def crop_face(img):
        """Detect face using Haar cascades and crop"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            raise ValueError("No faces detected.")
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    
    @staticmethod
    def process_face(img_path, target_size=112):
        """Complete face processing pipeline"""
        try:
            
            # Read image
            img = ImageProcessor.read_image(img_path)
            
            # Crop face
            # img = ImageProcessor.crop_face(img)
            
            
            # Resize keeping aspect ratio
            img = ImageProcessor.resize_image(img, target_size)
            
            # Align face
            # img = ImageProcessor.align_face(img)
                        
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

class TripletDataset(Dataset):
    def __init__(self, csv_path, sample_size=None):
        self.df = pd.read_csv(csv_path)
        if sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42)
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor = load_and_preprocess_image(row['Anchor'])
        positive = load_and_preprocess_image(row['Positive'])
        negative = load_and_preprocess_image(row['Negative'])
        
        if anchor is None or positive is None or negative is None:
            # Return first item if any image fails to load
            return self.__getitem__(0)
            
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }

def load_and_preprocess_image(image_path, target_size=112):
    """Load and preprocess image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize keeping aspect ratio
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Preprocess
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img
    except Exception as e:
        return None


def calculate_bhattacharyya_coefficient(dist1, dist2, bins=100):
    """Calculate Bhattacharyya coefficient between two distributions"""
    # Get histogram intersection
    hist1, bin_edges = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)
    
    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate Bhattacharyya coefficient
    bc = np.sum(np.sqrt(hist1 * hist2))
    
    return bc


def analyze_similarities_batch(csv_path, model_path, batch_size=128, sample_size=None):
    # Load model
    config = KinshipConfig()
    model = KinshipModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset and dataloader
    dataset = TripletDataset(csv_path, sample_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    # Lists to store similarities
    kin_similarities = []
    nonkin_similarities = []
    
    # Process batches
    print(f"Processing {len(dataset)} triplets in batches of {batch_size}")
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)
            
            # Extract features
            anchor_feat = model.feature_extractor(anchor)
            positive_feat = model.feature_extractor(positive)
            negative_feat = model.feature_extractor(negative)
            
            # Calculate similarities
            kin_sim = F.cosine_similarity(anchor_feat, positive_feat)
            nonkin_sim = F.cosine_similarity(anchor_feat, negative_feat)
            
            # Store results
            kin_similarities.extend(kin_sim.cpu().numpy())
            nonkin_similarities.extend(nonkin_sim.cpu().numpy())
    
    # Convert to numpy arrays
    kin_similarities = np.array(kin_similarities)
    nonkin_similarities = np.array(nonkin_similarities)
    
    # Calculate Bhattacharyya coefficient
    bc_kin_nonkin = calculate_bhattacharyya_coefficient(kin_similarities, nonkin_similarities)
    
    # Create output directory
    output_dir = '/cephyr/users/mehdiyev/Alvis/kinship_project/src/evaluations/cosine'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistics to file
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write("\nKin pairs statistics:\n")
        f.write(f"Count: {len(kin_similarities)}\n")
        f.write(f"Mean: {np.mean(kin_similarities):.4f}\n")
        f.write(f"Std: {np.std(kin_similarities):.4f}\n")
        f.write(f"Min: {np.min(kin_similarities):.4f}\n")
        f.write(f"Max: {np.max(kin_similarities):.4f}\n")
        
        f.write("\nNon-kin pairs statistics:\n")
        f.write(f"Count: {len(nonkin_similarities)}\n")
        f.write(f"Mean: {np.mean(nonkin_similarities):.4f}\n")
        f.write(f"Std: {np.std(nonkin_similarities):.4f}\n")
        f.write(f"Min: {np.min(nonkin_similarities):.4f}\n")
        f.write(f"Max: {np.max(nonkin_similarities):.4f}\n")
        
        f.write("\nBhattacharyya Coefficients:\n")
        f.write(f"Kin vs Non-kin: {bc_kin_nonkin:.4f}\n")
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=kin_similarities, label='Kin Pairs', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(data=nonkin_similarities, label='Non-kin Pairs', color='red', fill=True, alpha=0.3)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of Cosine Similarities between Kin and Non-kin Pairs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and plot ROC curve
    thresholds = np.linspace(0, 1, 100)
    tpr, fpr = [], []
    
    for threshold in thresholds:
        tp = np.sum(kin_similarities >= threshold)
        fn = np.sum(kin_similarities < threshold)
        fp = np.sum(nonkin_similarities >= threshold)
        tn = np.sum(nonkin_similarities < threshold)
        
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True, alpha=0.3)
    
    # Calculate AUC
    auc = np.trapz(tpr, fpr)
    plt.text(0.6, 0.2, f'AUC = {auc:.3f}', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate threshold metrics
    threshold_metrics = []
    for threshold in np.arange(0, 1.01, 0.05):
        tp = np.sum((kin_similarities >= threshold))
        tn = np.sum((nonkin_similarities < threshold))
        fp = np.sum((nonkin_similarities >= threshold))
        fn = np.sum((kin_similarities < threshold))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Save results
    threshold_df = pd.DataFrame(threshold_metrics)
    threshold_df.to_csv(os.path.join(output_dir, 'threshold_metrics.csv'), index=False)
    
    # Save similarity results
    results_df = pd.DataFrame({
        'kin_similarities': kin_similarities,
        'nonkin_similarities': nonkin_similarities
    })
    results_df.to_csv(os.path.join(output_dir, 'similarity_results.csv'), index=False)
    
    # Save best thresholds to file
    with open(os.path.join(output_dir, 'best_thresholds.txt'), 'w') as f:
        f.write("\nBest thresholds:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            best_idx = threshold_df[metric].idxmax()
            best_threshold = threshold_df.loc[best_idx, 'threshold']
            best_value = threshold_df.loc[best_idx, metric]
            f.write(f"Best {metric}: {best_value:.4f} at threshold {best_threshold:.2f}\n")
    
    return kin_similarities, nonkin_similarities


if __name__ == "__main__":
    # Paths
    csv_path = '/cephyr/users/mehdiyev/Alvis/kinship_project/data/processed/fiw/train/filtered_triplets_with_labels.csv'
    model_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model/model/best_kin_nonkin_model.pth'
    
    # Analyze similarities with batches
    kin_sims, nonkin_sims = analyze_similarities_batch(
        csv_path=csv_path,
        model_path=model_path,
        # sample_size=1000,
        batch_size=128
    )