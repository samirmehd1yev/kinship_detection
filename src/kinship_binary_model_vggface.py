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
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from facenet_pytorch import InceptionResnetV1

# Enable cuDNN benchmarking for better performance
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Config:
    def __init__(self):
        # Data paths
        self.train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/train_triplets_enhanced.csv'
        self.val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/val_triplets_enhanced.csv'
        self.test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/test_triplets_enhanced.csv'
        self.output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model'
        
        # Training settings
        self.batch_size = 64
        self.learning_rate = 5e-5
        self.weight_decay = 1e-6
        self.dropout_rate = 0.2
        self.num_workers = min(8, os.cpu_count())
        
        # Training schedule
        self.warmup_epochs = 1
        self.num_epochs = 15
        self.early_stopping_patience = 5
        self.scheduler_patience = 2
        
        # Model settings
        self.embedding_dim = 512
        self.margin = 0.3
        self.distance_threshold = 0.7 # check if this is good
        
        # Device settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Experiment settings
        self.exp_name = 'kinship_facenet_pretrained4'
        self.save_every = 1
        self.pin_memory = True
        
        self.setup_directories()
    
    def setup_directories(self):
        self.exp_dir = Path(self.output_dir) / self.exp_name
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        
        for dir_path in [self.exp_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        with open(self.exp_dir / 'config.json', 'w') as f:
            config_dict = {k: str(v) for k, v in vars(self).items() 
                          if not k.startswith('_') and isinstance(v, (str, int, float, bool))}
            json.dump(config_dict, f, indent=4)

class KinshipDatasetBinary(Dataset):
    def __init__(self, csv_path, transform=None, validation=False):
        self.data = pd.read_csv(csv_path)
        self.transform = transform or self.get_default_transforms(validation)
        self.validation = validation
        
        # Convert to pairs with labels
        self.pairs = []
        for _, row in self.data.iterrows():
            # Positive pair
            self.pairs.append({
                'img1': row['Anchor'],
                'img2': row['Positive'],
                'label': 1
            })
            # Negative pair
            self.pairs.append({
                'img1': row['Anchor'],
                'img2': row['Negative'],
                'label': 0
            })
        
        self.validate_data()
    
    @staticmethod
    def get_default_transforms(validation):
        if validation:
            return transforms.Compose([
                transforms.Resize((160, 160)),  # FaceNet expects 160x160
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def validate_data(self):
        valid_pairs = []
        print(f"Validating {len(self.pairs)} image pairs...")
        
        for pair in tqdm(self.pairs):
            if self.validate_pair(pair):
                valid_pairs.append(pair)
        
        self.pairs = valid_pairs
        print(f"Found {len(self.pairs)} valid pairs")
    
    def validate_pair(self, pair):
        return all(os.path.exists(p) for p in [pair['img1'], pair['img2']])
    
    def load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        img1 = self.load_image(pair['img1'])
        img2 = self.load_image(pair['img2'])
        
        if img1 is None or img2 is None:
            return self.__getitem__((idx + 1) % len(self))
        
        label = torch.tensor(pair['label'], dtype=torch.float32)
        
        return {
            'img1': img1,
            'img2': img2,
            'label': label,
            'paths': [pair['img1'], pair['img2']]
        }

class PretrainedFaceNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load pretrained InceptionResNetV1
        self.backbone = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            num_classes=None,
            device=config.device
        )
        
        # # Freeze early layers
        # for name, param in self.backbone.named_parameters():
        #     if 'block8' not in name and 'mixed_7a' not in name:
        #         param.requires_grad = False
        
        # Add custom layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(512, config.embedding_dim)
        self.bn = nn.BatchNorm1d(config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.layer_norm(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class KinshipVerificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = PretrainedFaceNetModel(config)
        self.distance_threshold = config.distance_threshold
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, img1, img2):
        with torch.no_grad():
            emb1 = self.forward(img1)
            emb2 = self.forward(img2)
            cosine_sim = F.cosine_similarity(emb1, emb2)
            cosine_dist = 1 - cosine_sim
            prediction = (cosine_dist < self.distance_threshold).float()
            return prediction, cosine_dist
    
    def predict_proba(self, img1, img2):
        with torch.no_grad():
            emb1 = self.forward(img1)
            emb2 = self.forward(img2)
            cosine_sim = F.cosine_similarity(emb1, emb2)
            prob = (cosine_sim + 1) / 2
            return prob

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # Embeddings are already normalized in the model
        cosine_similarity = F.cosine_similarity(output1, output2)
        cosine_distance = 1 - cosine_similarity
        
        pos_loss = label * torch.pow(cosine_distance, 2)
        neg_loss = (1 - label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
        
        loss = torch.mean(pos_loss + neg_loss)
        
        # Store metrics
        self.pos_dist_mean = cosine_distance[label == 1].mean().item() if torch.any(label == 1) else 0
        self.neg_dist_mean = cosine_distance[label == 0].mean().item() if torch.any(label == 0) else 0
        
        return loss

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch):
    model.train()
    scaler = GradScaler()
    running_loss = 0.0
    running_acc = 0.0
    num_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
    for batch_idx, batch in enumerate(pbar):
        img1 = batch['img1'].to(config.device, non_blocking=True)
        img2 = batch['img2'].to(config.device, non_blocking=True)
        label = batch['label'].to(config.device, non_blocking=True)
        
        batch_size = img1.size(0)
        num_samples += batch_size
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, label)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(output1, output2)
            cosine_dist = 1 - cosine_sim
            predictions = (cosine_dist < model.distance_threshold).float()
            correct = (predictions == label).float().sum()
            
            running_loss += loss.item() * batch_size
            running_acc += correct.item()
        
        avg_loss = running_loss / num_samples
        avg_acc = running_acc / num_samples
        
        pbar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'acc': f"{avg_acc:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    if scheduler is not None:
        scheduler.step()
    
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
            img1 = batch['img1'].to(config.device, non_blocking=True)
            img2 = batch['img2'].to(config.device, non_blocking=True)
            label = batch['label'].to(config.device, non_blocking=True)
            
            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, label)
            
            cosine_sim = F.cosine_similarity(output1, output2)
            cosine_dist = 1 - cosine_sim
            predictions = (cosine_dist < config.distance_threshold).float()
            correct = (predictions == label).float().sum()
            
            batch_size = img1.size(0)
            num_samples += batch_size
            running_loss += loss.item() * batch_size
            running_acc += correct.item()
    
    return {
        'val_loss': running_loss / num_samples,
        'val_accuracy': running_acc / num_samples
    }

def test_model(model, test_loader, config):
    model.eval()
    running_acc = 0.0
    num_samples = 0
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            img1 = batch['img1'].to(config.device, non_blocking=True)
            img2 = batch['img2'].to(config.device, non_blocking=True)
            label = batch['label'].to(config.device, non_blocking=True)
            
            output1 = model(img1)
            output2 = model(img2)
            
            cosine_sim = F.cosine_similarity(output1, output2)
            cosine_dist = 1 - cosine_sim
            predictions = (cosine_dist < config.distance_threshold).float()
            correct = (predictions == label).float().sum()
            
            batch_size = img1.size(0)
            num_samples += batch_size
            running_acc += correct.item()
            
            all_distances.extend(cosine_dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    accuracy = running_acc / num_samples
    auc = calculate_auc(all_distances, all_labels)
    
    metrics = {
        'test_accuracy': accuracy,
        'test_auc': auc,
        'mean_distance': np.mean(all_distances),
        'std_distance': np.std(all_distances)
    }
    
    return metrics

def calculate_auc(distances, labels):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, -np.array(distances))

class EarlyStopping:
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

def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False, best_val_loss=None, wandb_run_id=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'best_val_loss': best_val_loss,
        'wandb_run_id': wandb_run_id
    }
    
    checkpoint_path = config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = config.checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def main():
    # Initialize config and set seed
    config = Config()
    set_seed(42)
    
    # Check for existing checkpoint
    best_model_path = config.checkpoint_dir / 'best_model.pth'
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Initialize wandb with resume capability
    run_id = None
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        if 'wandb_run_id' in checkpoint:
            run_id = checkpoint['wandb_run_id']
    
    wandb.init(
        project="kinship-verification-pretrained",
        name=config.exp_name,
        config=vars(config),
        id=run_id,
        resume="allow"
    )
    
    # Create datasets and dataloaders
    train_dataset = KinshipDatasetBinary(config.train_path, validation=False)
    val_dataset = KinshipDatasetBinary(config.val_path, validation=True)
    test_dataset = KinshipDatasetBinary(config.test_path, validation=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
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
    
    # Initialize model, criterion
    model = KinshipVerificationModel(config).to(config.device)
    criterion = ContrastiveLoss(margin=config.margin)
    
    # Separate parameters for different learning rates
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone.backbone' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    # Configure optimizer
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': config.learning_rate * 0.1},
        {'params': new_params, 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
    
    # Load checkpoint if exists
    if best_model_path.exists():
        print(f"Loading checkpoint from {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.learning_rate * 0.1, config.learning_rate],
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1e4
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, config, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config)
        
        # Log metrics
        wandb.log({
            **train_metrics,
            **val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        })
        
        # Save checkpoint if best model
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                is_best=True,
                best_val_loss=best_val_loss,
                wandb_run_id=wandb.run.id
            )
        
        # Regular checkpoint saving
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                best_val_loss=best_val_loss,
                wandb_run_id=wandb.run.id
            )
        
        # Print metrics
        print(f"Train Loss: {train_metrics['train_loss']:.4f} "
              f"Train Acc: {train_metrics['train_accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f} "
              f"Val Acc: {val_metrics['val_accuracy']:.4f}")
        
        # Early stopping check
        if early_stopping(val_metrics['val_loss']):
            print("Early stopping triggered!")
            break
    
    # Final evaluation on test set
    print("\nPerforming final evaluation on test set...")
    test_metrics = test_model(model, test_loader, config)
    wandb.log(test_metrics)
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    wandb.finish()
if __name__ == "__main__":
    main()