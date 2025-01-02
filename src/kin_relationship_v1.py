import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import json
from torchvision import transforms
import onnx
from onnx2torch import convert
import wandb
from tqdm import tqdm
import numpy as np
import os
import functools
from torch.optim.lr_scheduler import OneCycleLR

# Set GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print = functools.partial(print, flush=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define relationship types and mappings
RELATIONSHIP_TYPES = ['ms', 'md', 'fs', 'fd', 'ss', 'bb', 'sibs']
REL_TO_IDX = {rel: idx for idx, rel in enumerate(RELATIONSHIP_TYPES)}
IDX_TO_REL = {idx: rel for rel, idx in REL_TO_IDX.items()}

# Gender-based relationship constraints
VALID_RELATIONSHIPS = {
    # (gender1, gender2) -> valid relationships
    (0, 0): ['md', 'ss'],           # Female-Female
    (1, 1): ['fs', 'bb'],           # Male-Male
    (0, 1): ['ms', 'fd', 'sibs'],   # Female-Male
    (1, 0): ['ms', 'fd', 'sibs']    # Male-Female
}

class GenderAwareCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            # note: Change for class imbalance(default is 1.0 for all classes) 
            class_weights = torch.tensor([
                1.0,  # ms
                1.0,  # md
                1.0,  # fs
                1.0,  # fd
                1.0,  # ss 
                1.0,  # bb 
                1.0   # sibs
            ])
        self.class_weights = class_weights
    
    def forward(self, logits, labels, gender_features):
        # Apply sigmoid to stabilize logits
        logits = F.log_softmax(logits, dim=1)
        
        # Apply class weights
        weights = self.class_weights.to(logits.device)[labels]
        
        # Normalized cross-entropy loss with better numerical stability
        ce_loss = F.nll_loss(logits, labels, reduction='none')
        weighted_loss = weights * ce_loss
        
        # Reduced gender penalty to prevent loss explosion
        gender1, gender2 = gender_features[:, 0], gender_features[:, 1]
        pred_labels = logits.argmax(dim=1)
        
        gender_penalty = torch.zeros_like(ce_loss)
        
        for i in range(len(labels)):
            valid_rels = VALID_RELATIONSHIPS[(int(gender1[i].item()), int(gender2[i].item()))]
            valid_indices = torch.tensor([REL_TO_IDX[rel] for rel in valid_rels], 
                                       device=logits.device)
            
            if pred_labels[i] not in valid_indices:
                gender_penalty[i] = 5.0  # Reduced penalty
        
        total_loss = weighted_loss + gender_penalty
        
        # Add gradient clipping within the loss
        torch.nn.utils.clip_grad_norm_(total_loss, max_norm=1.0)
        
        return total_loss.mean()
    
class RelationshipClassifier(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        print(f"Loading ONNX model from: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        self.backbone = convert(onnx_model)
        self.backbone.requires_grad_(True)
        print("ONNX model loaded and converted successfully")
        
        self.embedding_dim = 512
        self.gender_dim = 2
        self.num_classes = len(RELATIONSHIP_TYPES)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + self.gender_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.ModuleDict({
                'layer': nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            }),
            nn.ModuleDict({
                'layer': nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            })
        ])
        
        # Final layers
        self.final_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Linear(256, self.num_classes)
    
    def forward_one(self, x):
        return self.backbone(x)
    
    def forward(self, x1, x2, gender_features):
        # Get embeddings
        emb1 = F.normalize(self.forward_one(x1), p=2, dim=1)
        emb2 = F.normalize(self.forward_one(x2), p=2, dim=1)
        
        # Concatenate features
        combined = torch.cat([emb1, emb2, gender_features], dim=1)
        
        # Process through network
        features = self.input_proj(combined)
        
        for block in self.residual_blocks:
            residual = features
            features = block['layer'](features)
            features = features + residual
        
        features = self.final_proj(features)
        logits = self.classifier(features)
        
        # Apply gender-based masking
        gender1, gender2 = gender_features[:, 0], gender_features[:, 1]
        mask = torch.full_like(logits, float('-inf'))
        
        for i in range(len(gender1)):
            valid_rels = VALID_RELATIONSHIPS[(int(gender1[i].item()), int(gender2[i].item()))]
            valid_indices = [REL_TO_IDX[rel] for rel in valid_rels]
            mask[i, valid_indices] = 0
        
        return logits + mask

class RelationshipDataset(Dataset):
    def __init__(self, df, gender_dict, transform=None, is_training=True):
        self.df = df
        self.gender_dict = gender_dict
        
        for col in ['Anchor', 'Positive', 'Negative']:
            self.df[col] = self.df[col].str.replace(
                'data/',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/',
                regex=False
            )
        
        # Add error tracking
        self.load_errors = []
        
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transform = transform
    
    def load_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(img)
        except Exception as e:
            self.load_errors.append({
                'path': image_path,
                'error': str(e)
            })
            raise e
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Load images with error handling
            try:
                anchor_img = self.load_image(row['Anchor'])
                positive_img = self.load_image(row['Positive'])
            except Exception as e:
                print(f"Error loading images for index {idx}: {str(e)}")
                # Skip to next valid sample
                if idx + 1 < len(self):
                    return self.__getitem__(idx + 1)
                else:
                    raise RuntimeError("No valid samples found")
            
            # Get relationship label
            rel_type = row['ptype']
            label = torch.tensor(REL_TO_IDX[rel_type])
            
            # Get gender features with validation
            anchor_gender = self.gender_dict.get(row['Anchor'])
            positive_gender = self.gender_dict.get(row['Positive'])
            
            if anchor_gender is None or positive_gender is None:
                print(f"Warning: Missing gender information for index {idx}")
                anchor_gender = 0 if anchor_gender is None else anchor_gender
                positive_gender = 0 if positive_gender is None else positive_gender
            
            gender_features = torch.tensor([anchor_gender, positive_gender], dtype=torch.float32)
            
            return {
                'anchor': anchor_img,
                'positive': positive_img,
                'gender_features': gender_features,
                'label': label,
                'metadata': {
                    'rel_type': rel_type,
                    'anchor_path': row['Anchor'],
                    'positive_path': row['Positive']
                }
            }
        except Exception as e:
            print(f"Error processing sample at index {idx}: {str(e)}")
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            raise RuntimeError("No valid samples found in the dataset")

def train_model(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = GenderAwareCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=config['learning_rate'], 
                          epochs=config['epochs'], steps_per_epoch=len(train_loader))
    
    best_val_acc = 0
    patience = 0
    max_patience = 3
    
    # Add error tracking
    training_errors = []
    
    try:
        for epoch in range(config['epochs']):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            gender_violations = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Ensure batch size is greater than 1 for batch norm
                    if len(batch['anchor']) <= 1:
                        continue
                    
                    # Separate metadata from tensors
                    metadata = batch.pop('metadata')
                    # Move tensors to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    optimizer.zero_grad()
                    
                    # Add gradient computation error handling
                    try:
                        logits = model(batch['anchor'], batch['positive'], batch['gender_features'])
                        loss = criterion(logits, batch['label'], batch['gender_features'])
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            print(f"CUDA out of memory in batch {batch_idx}. Skipping...")
                            continue
                        else:
                            raise e
                    
                    # Update metrics
                    pred = logits.argmax(dim=1)
                    correct += (pred == batch['label']).sum().item()
                    total += len(batch['label'])
                    train_loss += loss.item()
                    
                    # Check gender violations
                    for i in range(len(pred)):
                        gender1, gender2 = batch['gender_features'][i]
                        valid_rels = VALID_RELATIONSHIPS[(int(gender1.item()), int(gender2.item()))]
                        pred_rel = IDX_TO_REL[pred[i].item()]
                        if pred_rel not in valid_rels:
                            gender_violations += 1
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{correct/total:.4f}",
                        'viol': f"{gender_violations/total:.4f}"
                    })
                
                except Exception as e:
                    error_info = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'error': str(e)
                    }
                    training_errors.append(error_info)
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Validation
            try:
                val_metrics = evaluate_model(model, val_loader, criterion, device)
                
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}")
                print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_val_acc': best_val_acc,
                        }, 'checkpoints/kin_relationship_v1/best_model.pth')
                    except Exception as e:
                        print(f"Error saving checkpoint: {str(e)}")
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        print("Early stopping triggered")
                        break
                        
            except Exception as e:
                print(f"Error during validation: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
    
    finally:
        # Save error log
        if training_errors:
            with open('training_errors.json', 'w') as f:
                json.dump(training_errors, f, indent=2)
    
    return best_val_acc

def evaluate_model(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    gender_violations = 0
    per_class_correct = {rel: 0 for rel in RELATIONSHIP_TYPES}
    per_class_total = {rel: 0 for rel in RELATIONSHIP_TYPES}
    confusion_matrix = np.zeros((len(RELATIONSHIP_TYPES), len(RELATIONSHIP_TYPES)))
    
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            metadata = batch.pop('metadata')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            logits = model(batch['anchor'], batch['positive'], batch['gender_features'])
            loss = criterion(logits, batch['label'], batch['gender_features'])
            
            pred = logits.argmax(dim=1)
            correct += (pred == batch['label']).sum().item()
            total += len(batch['label'])
            total_loss += loss.item()
            
            for i, (p, t) in enumerate(zip(pred, batch['label'])):
                # Update confusion matrix
                confusion_matrix[t.item()][p.item()] += 1
                
                # Per-class accuracy
                true_rel = IDX_TO_REL[t.item()]
                pred_rel = IDX_TO_REL[p.item()]
                per_class_total[true_rel] += 1
                if p == t:
                    per_class_correct[true_rel] += 1
                
                # Check gender violations
                gender1, gender2 = batch['gender_features'][i]
                valid_rels = VALID_RELATIONSHIPS[(int(gender1.item()), int(gender2.item()))]
                
                if pred_rel not in valid_rels:
                    gender_violations += 1
                
                # Record errors
                if p != t or pred_rel not in valid_rels:
                    all_errors.append({
                        'true_rel': true_rel,
                        'pred_rel': pred_rel,
                        'genders': (gender1.item(), gender2.item()),
                        'valid_rels': valid_rels,
                        'is_gender_violation': pred_rel not in valid_rels,
                        'anchor_path': metadata['anchor_path'][i],
                        'positive_path': metadata['positive_path'][i]
                    })
    
    per_class_acc = {rel: per_class_correct[rel]/per_class_total[rel] 
                     if per_class_total[rel] > 0 else 0 
                     for rel in RELATIONSHIP_TYPES}
    
    # Save error analysis
    error_file = 'evaluation_errors.json'
    with open(error_file, 'w') as f:
        json.dump({
            'summary': {
                'total_samples': total,
                'accuracy': correct/total,
                'gender_violation_rate': gender_violations/total,
                'per_class_accuracy': per_class_acc
            },
            'confusion_matrix': confusion_matrix.tolist(),
            'errors': all_errors
        }, f, indent=2)
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / total,
        'per_class_acc': per_class_acc,
        'per_class_total': per_class_total,
        'gender_violation_rate': gender_violations / total,
        'confusion_matrix': confusion_matrix,
        'errors': all_errors
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'analyze'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }
    
    # Data loading
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train'
    splits_to_load = ['test'] if args.mode == 'test' else ['train', 'val', 'test']
    
    # Load datasets directly
    print("\nLoading datasets...")
    data_splits = {}
    for split in splits_to_load:
        csv_path = f"{base_path}/splits_no_overlap_hand/{split}_triplets_enhanced.csv"
        data_splits[split] = pd.read_csv(csv_path)
        print(f"Loaded {split} split: {len(data_splits[split])} samples")
    
    # Load gender information
    metadata_path = os.path.join(base_path, 'fiw_metadata_filtered.csv')
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[~metadata_df['Aligned_Image_Path'].str.contains('unrelated')]
    
    for col in ['Aligned_Image_Path']:
        metadata_df[col] = metadata_df[col].str.replace(
            'data/',
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/',
            regex=False
        )
    gender_dict = {
        row['Aligned_Image_Path']: 1 if row['True_Gender'].lower() == 'm' else 0
        for _, row in metadata_df.iterrows()
    }
    
    # Create datasets and dataloaders
    datasets = {
        split: RelationshipDataset(data_splits[split], gender_dict, is_training=(split=='train'))
        for split in splits_to_load
    }
    
    dataloaders = {
        split: DataLoader(datasets[split], 
                         batch_size=config['batch_size'],
                         shuffle=(split == 'train'),
                         num_workers=4,
                         pin_memory=True,
                         drop_last=True)
        for split in splits_to_load
    }
    
    # Model initialization
    print("\nInitializing model...")
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    model = RelationshipClassifier(onnx_path)
    
    if args.mode == 'train':
        print("\nStarting training...")
        best_val_acc = train_model(model, dataloaders['train'], dataloaders['val'], config)
        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    
    elif args.mode == 'test':
        print("\nLoading best model for testing...")
        checkpoint = torch.load('checkpoints/kin_relationship_v1/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        criterion = GenderAwareCrossEntropyLoss()
        test_metrics = evaluate_model(model, dataloaders['test'], criterion, device)
        
        print("\nTest Results:")
        print(f"Overall Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Gender Violation Rate: {test_metrics['gender_violation_rate']:.4f}")
        print("\nPer-Class Accuracy:")
        for rel, acc in test_metrics['per_class_acc'].items():
            print(f"{rel}: {acc:.4f} ({test_metrics['per_class_total'][rel]} samples)")
    
    elif args.mode == 'analyze':
        checkpoint = torch.load('checkpoints/kin_relationship_v1/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        criterion = GenderAwareCrossEntropyLoss()
        
        print("\nAnalyzing errors on test set...")
        test_metrics = evaluate_model(model, dataloaders['test'], criterion, device)
        
        # Print most common error patterns
        errors = test_metrics['errors']
        error_patterns = {}
        for error in errors:
            key = (error['true_rel'], error['pred_rel'], error['is_gender_violation'])
            if key not in error_patterns:
                error_patterns[key] = []
            error_patterns[key].append(error)
        
        print("\nMost Common Error Patterns:")
        for (true_rel, pred_rel, is_violation), examples in sorted(
            error_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"\n{true_rel} -> {pred_rel} (Gender violation: {is_violation})")
            print(f"Total occurrences: {len(examples)}")
            print("Example cases:")
            for example in examples[:3]:
                print(f"- Genders: {example['genders']}")
                print(f"  Valid relationships: {example['valid_rels']}")
                print(f"  Anchor: {example['anchor_path']}")
                print(f"  Positive: {example['positive_path']}")

if __name__ == '__main__':
    main()
    
# (kinship_venv_insightface) [mehdiyev@alvis1 src]$ python kin_relationship_v1.py --mode test
# Using device: cuda

# Loading datasets...
# Loaded test split: 26229 samples

# Initializing model...
# Loading ONNX model from: /cephyr/users/mehdiyev/Alvis/.insightface/models/buffalo_l/w600k_r50.onnx
# ONNX model loaded and converted successfully

# Loading best model for testing...
# /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/src/kin_relationship_v1.py:566: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   checkpoint = torch.load('checkpoints/kin_relationship_v1/best_model.pth')
# Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 819/819 [02:09<00:00,  6.31it/s]

# Test Results:
# Overall Accuracy: 0.8628
# Gender Violation Rate: 0.0000

# Per-Class Accuracy:
# ms: 0.9373 (3875 samples)
# md: 0.9486 (4827 samples)
# fs: 0.9607 (4632 samples)
# fd: 0.9552 (5285 samples)
# ss: 0.6545 (2802 samples)
# bb: 0.6493 (1999 samples)
# sibs: 0.6352 (2788 samples)