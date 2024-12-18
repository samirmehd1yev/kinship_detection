import os
import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import onnx
from onnx2torch import convert
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch.nn as nn

class KinshipVerificationModel(torch.nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        onnx_model = onnx.load(onnx_path)
        self.backbone = convert(onnx_model)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
    
    def forward_one(self, x):
        emb = self.backbone(x)
        return emb
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

class KinFaceWDataset(Dataset):
    """Dataset class for KinFaceW-I and KinFaceW-II"""
    def __init__(self, base_path, relation, dataset_type='KinFaceW-I'):
        self.base_path = base_path
        self.relation = relation
        self.dataset_type = dataset_type
        self.transform = self.get_transforms()
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Map relation names to folder names
        self.relation_map = {
            'fs': 'father-son',
            'fd': 'father-dau',
            'ms': 'mother-son',
            'md': 'mother-dau'
        }
        
        # Load meta data
        meta_path = os.path.join(base_path, dataset_type, 'meta_data', f'{relation}_pairs.mat')
        meta_data = scipy.io.loadmat(meta_path)
        print(f"Loaded meta data for {relation} relation in {dataset_type}")
        print(f"Meta data keys: {meta_data.keys()}")
        
        # Get pairs data
        pairs_data = meta_data['pairs']
        
        # Extract labels and image paths
        self.labels = np.array([item[0][0] for item in pairs_data[:, 1]]).astype(np.float32)
        img1_paths = [item[0] for item in pairs_data[:, 2]]
        img2_paths = [item[0] for item in pairs_data[:, 3]]
        
        # Create list of pairs
        self.pairs = list(zip(img1_paths, img2_paths))
        
        print(f"Loaded {len(self.pairs)} pairs for {relation} relation in {dataset_type}")

    def get_transforms(self):
        """Get image transforms matching the training transforms"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def prepare_face(self, image_path):
        """Process a single face from an image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image file: {image_path}")
            
            # Detect faces
            faces = self.face_app.get(img)
            
            if not faces:
                raise ValueError(f"No faces detected in image: {image_path}")
            
            if len(faces) > 1:
                print(f"Multiple faces detected in {image_path}. Using the largest face.")
            
            # Select largest face
            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            
            # Get keypoints and align face
            kps = face.kps.astype(int)
            aligned_face = face_align.norm_crop(img, kps)
            
            if aligned_face is None or aligned_face.size == 0:
                raise ValueError("Face alignment failed")
            
            # Convert color space
            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            
            return aligned_face
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            raise

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            img1_name, img2_name = self.pairs[idx]
            
            # Construct full image paths
            img1_path = os.path.join(
                self.base_path,
                self.dataset_type,
                'images',
                self.relation_map[self.relation],
                img1_name
            )
            img2_path = os.path.join(
                self.base_path,
                self.dataset_type,
                'images',
                self.relation_map[self.relation],
                img2_name
            )
            
            # Load and align faces
            img1 = self.prepare_face(img1_path)
            img2 = self.prepare_face(img2_path)
            
            # Apply transforms
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            return {
                'image1': img1,
                'image2': img2,
                'label': torch.tensor(self.labels[idx], dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"Error loading pair {idx}: {str(e)}")
            print(f"Image paths: \n{img1_path}\n{img2_path}")
            raise

def custom_collate(batch):
    """Custom collate function to handle None values"""
    return {
        'image1': torch.stack([item['image1'] for item in batch]),
        'image2': torch.stack([item['image2'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }

def evaluate_model_on_kinfacew(model, device, base_path, output_dir, threshold=None):
    """Evaluate model on KinFaceW datasets"""
    model.eval()
    dataset_types = ['KinFaceW-I', 'KinFaceW-II']
    relations = ['fs', 'fd', 'ms', 'md']
    
    all_results = {}
    
    for dataset_type in dataset_types:
        print(f"\nEvaluating on {dataset_type}...")
        dataset_results = {}
        
        for relation in relations:
            print(f"\nTesting {relation} relation...")
            
            # Create dataset
            dataset = KinFaceWDataset(
                base_path=base_path,
                relation=relation,
                dataset_type=dataset_type
            )
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=custom_collate
            )
            
            # Evaluation variables
            all_preds = []
            all_labels = []
            all_similarities = []
            
            # Evaluate
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Evaluating {relation}"):
                    image1 = batch['image1'].to(device)
                    image2 = batch['image2'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Get embeddings
                    emb1, emb2 = model(image1, image2)
                    
                    # Normalize embeddings
                    emb1 = F.normalize(emb1, p=2, dim=1)
                    emb2 = F.normalize(emb2, p=2, dim=1)
                    
                    # Calculate cosine similarity
                    similarity = F.cosine_similarity(emb1, emb2)
                    
                    # Use provided threshold or default to 0.5
                    current_threshold = threshold if threshold is not None else 0.5
                    preds = (similarity > current_threshold).float()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_similarities.extend(similarity.cpu().numpy())
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'precision': precision_score(all_labels, all_preds),
                'recall': recall_score(all_labels, all_preds),
                'f1': f1_score(all_labels, all_preds),
                'auc': roc_auc_score(all_labels, all_similarities)
            }
            
            dataset_results[relation] = metrics
            
            print(f"\nResults for {relation}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        all_results[dataset_type] = dataset_results
    
    # Save results
    save_evaluation_results(all_results, output_dir)
    return all_results

def save_evaluation_results(results, output_dir):
    """Save evaluation results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"evaluation_results_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save detailed results as JSON
    with open(os.path.join(output_path, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create summary DataFrame
    summary_data = []
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for dataset_type in results:
        for relation in results[dataset_type]:
            relation_metrics = results[dataset_type][relation]
            
            summary_row = {
                'dataset': dataset_type,
                'relation': relation,
                **{metric: relation_metrics[metric] for metric in metrics}
            }
            summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, "summary_results.csv"), index=False)
    
    # Print summary
    print(f"\nResults saved to: {output_path}")
    print("\nSummary of results:")
    print(summary_df.to_string())

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get ONNX model path
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    
    # Initialize model
    model = KinshipVerificationModel(onnx_path)
    
    # Load the trained model weights from checkpoint
    checkpoint_path = 'checkpoints/kin_binary_v1/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get best threshold from checkpoint
    best_threshold = checkpoint.get('best_threshold', 0.5)
    print(f"Using threshold: {best_threshold}")
    
    # Set paths
    base_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data"
    output_dir = "evaluations/results_kinfacew"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate on KinFaceW datasets
    results = evaluate_model_on_kinfacew(model, device, base_path, output_dir, threshold=best_threshold)

if __name__ == "__main__":
    main()