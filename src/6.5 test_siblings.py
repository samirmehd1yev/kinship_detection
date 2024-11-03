import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import cv2
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
import random

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

class ImageProcessor:
    @staticmethod
    def read_image(path):
        """Read image using OpenCV and convert to RGB"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def resize_image(img, size):
        """Resize image to target size"""
        return cv2.resize(img, (size, size))
    
    @staticmethod
    def process_face(img_path, target_size=112):
        """Process face image - just resize for SiblingsDB as faces are already detected"""
        try:
            # Read image
            img = ImageProcessor.read_image(img_path)
            
            # Resize to target size
            img = ImageProcessor.resize_image(img, target_size)
            
            # Convert to float and normalize
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) * 2
            
            # Convert to tensor with correct channel order
            img = torch.from_numpy(img.transpose(2, 0, 1))
            
            return img
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None

class LQfDataset(Dataset):
    """Dataset class specifically for LQf dataset with case-insensitive filename matching"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
        # Load and display first few rows of CSV
        csv_path = self.base_path / 'LQf.csv'
        self.pairs_df = pd.read_csv(csv_path)
        print("\nFirst few rows of LQf.csv:")
        print(self.pairs_df.head())
        print("\nColumns:", self.pairs_df.columns.tolist())
        
        # Get image directory and list contents
        self.img_dir = os.path.join(base_path, 'DBs', 'LQf')
        print(f"\nChecking image directory: {self.img_dir}")
        if os.path.exists(self.img_dir):
            print("Directory exists.")
            # Create a case-insensitive filename mapping
            self.filename_map = self._create_filename_map()
            print("Sample contents:", list(self.filename_map.values())[:10])
        else:
            raise ValueError(f"Image directory not found: {self.img_dir}")
        
        # Extract image paths and labels
        self.pairs = []
        self.labels = []
        
        processed = 0
        errors = 0
        
        for idx, row in self.pairs_df.iterrows():
            try:
                # Each row should have two person names and a label
                person1, person2, label = row.iloc[0:3].values
                
                # Get the actual filenames using case-insensitive matching
                img1_path = self._find_image_path(person1.strip())
                img2_path = self._find_image_path(person2.strip())
                
                # Debug first few pairs
                if idx < 5:
                    print(f"\nPair {idx}:")
                    print(f"Person 1: {person1} -> {img1_path}")
                    print(f"Person 2: {person2} -> {img2_path}")
                    print(f"Label: {label}")
                    print(f"Files exist: {os.path.exists(img1_path) if img1_path else False} and {os.path.exists(img2_path) if img2_path else False}")
                
                # Verify files exist
                if not img1_path or not os.path.exists(img1_path):
                    raise FileNotFoundError(f"Image not found for person: {person1}")
                if not img2_path or not os.path.exists(img2_path):
                    raise FileNotFoundError(f"Image not found for person: {person2}")
                
                self.pairs.append((img1_path, img2_path))
                # Convert -1 to 0 for binary classification
                self.labels.append(1 if int(label) == 1 else 0)
                processed += 1
                
            except Exception as e:
                errors += 1
                if errors < 10:  # Only show first 10 errors
                    print(f"Error processing row {idx}: {str(e)}")
                continue
        
        print(f"\nLQf Dataset Summary:")
        print(f"Successfully processed {processed} pairs")
        print(f"Encountered {errors} errors")

    def _create_filename_map(self):
        """Create a mapping of lowercase filenames to actual filenames"""
        filename_map = {}
        for filename in os.listdir(self.img_dir):
            # Only map image files
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                filename_map[filename.split('.')[0].lower()] = filename
        return filename_map

    def _find_image_path(self, person_name):
        """Find the actual image path using case-insensitive matching"""
        # Convert person name to lowercase for matching
        person_name_lower = person_name.lower()
        
        # Try to find the matching filename
        if person_name_lower in self.filename_map:
            return os.path.join(self.img_dir, self.filename_map[person_name_lower])
        
        # Try some common variations if the exact match is not found
        # This handles cases like 'limaFilho' vs 'limafilho'
        variations = [
            person_name_lower,
            person_name_lower.replace('_', ''),
            person_name.replace('_', '')
        ]
        
        for name in variations:
            # Try to find a partial match
            matches = [filename for key, filename in self.filename_map.items() 
                      if name in key or key in name]
            if matches:
                return os.path.join(self.img_dir, matches[0])
        
        return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        try:
            # Process images
            img1 = ImageProcessor.process_face(img1_path)
            img2 = ImageProcessor.process_face(img2_path)
            
            if img1 is None or img2 is None:
                raise ValueError("Failed to process one or both images")
            
            return {
                'img1': img1,
                'img2': img2,
                'label': torch.tensor(label, dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {str(e)}")
            return {
                'img1': torch.zeros(3, 112, 112),
                'img2': torch.zeros(3, 112, 112),
                'label': torch.tensor(label, dtype=torch.float32)
            }


def evaluate_dataset(model, dataset, device, batch_size=32):
    """Evaluate model on dataset"""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating LQf"):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(img1, img2)
            probs = torch.sigmoid(outputs['kinship_score'])
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    try:
        # Calculate binary classification metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='binary', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs)
        }
        
        # Add class distribution information
        class_counts = pd.Series(all_labels).value_counts().to_dict()
        metrics['class_distribution'] = {
            'kin (1)': int(class_counts.get(1, 0)),
            'non-kin (0)': int(class_counts.get(0, 0))
        }
        
        # Calculate confusion matrix counts
        tn = np.sum((all_labels == 0) & (all_preds == 0))
        fp = np.sum((all_labels == 0) & (all_preds == 1))
        fn = np.sum((all_labels == 1) & (all_preds == 0))
        tp = np.sum((all_labels == 1) & (all_preds == 1))
        
        metrics['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
            
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        metrics = {
            'error': str(e),
            'class_distribution': pd.Series(all_labels).value_counts().to_dict()
        }
    
    return metrics

def save_results(results, output_dir):
    """Save evaluation results with enhanced formatting"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"lqf_results_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Format results for display
    display_results = {}
    for k, v in results.items():
        if isinstance(v, float):
            display_results[k] = f"{v:.4f}"
        else:
            display_results[k] = v
    
    # Save as JSON with proper formatting
    with open(os.path.join(output_path, "results.json"), 'w') as f:
        json.dump(display_results, f, indent=4)
    
    # Save as CSV with proper handling of nested structures
    flat_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_results[f"{k}_{sub_k}"] = sub_v
        else:
            flat_results[k] = v
    
    pd.DataFrame([flat_results]).to_csv(os.path.join(output_path, "results.csv"), index=False)
    
    # Print summary
    print("\nResults saved to:", output_path)
    print("\nResults summary:")
    
    # Print metrics first
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        if metric in display_results:
            print(f"{metric}: {display_results[metric]}")
    
    # Print class distribution
    if 'class_distribution' in display_results:
        print("\nClass distribution:")
        for cls, count in display_results['class_distribution'].items():
            print(f"  {cls}: {count}")
    
    # Print confusion matrix
    if 'confusion_matrix' in display_results:
        print("\nConfusion Matrix:")
        cm = display_results['confusion_matrix']
        print(f"  True Negative: {cm['true_negative']}")
        print(f"  False Positive: {cm['false_positive']}")
        print(f"  False Negative: {cm['false_negative']}")
        print(f"  True Positive: {cm['true_positive']}")
        
def main():
    """Main function for evaluating kinship model on LQf dataset"""
    try:
        # Configuration and setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        # Model setup
        print("\nInitializing model...")
        config = KinshipConfig()
        model = KinshipModel(config).to(device)
        
        # Load trained model
        model_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model/model/best_kin_nonkin_model.pth"
        print(f"\nLoading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
        
        # Paths
        base_path = "/cephyr/users/mehdiyev/Alvis/kinship_project/data/SiblingsDB"
        output_dir = "/cephyr/users/mehdiyev/Alvis/kinship_project/src/evaluations/results_siblings"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nInitializing dataset...")
        try:
            # Create LQf dataset
            dataset = LQfDataset(base_path)
            
            if len(dataset) == 0:
                print("Error: No valid pairs found in LQf dataset")
                return
            
            print(f"\nDataset loaded successfully with {len(dataset)} pairs")
            
            # Evaluate model
            print("\nStarting evaluation...")
            results = evaluate_dataset(model, dataset, device)
            
            # Save and display results
            save_results(results, output_dir)
            
            # Print additional statistics
            if 'class_distribution' in results:
                total_pairs = sum(results['class_distribution'].values())
                print(f"\nTotal evaluated pairs: {total_pairs}")
                
                if 'confusion_matrix' in results:
                    cm = results['confusion_matrix']
                    total_correct = cm['true_positive'] + cm['true_negative']
                    print(f"Correctly classified pairs: {total_correct}")
                    print(f"Misclassified pairs: {total_pairs - total_correct}")
            
        except Exception as e:
            print(f"Error processing LQf dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run main function
    main()