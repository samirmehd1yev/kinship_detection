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

# Reuse your existing imports and model classes from the original code
from kin_nonkin_test import KinshipConfig, KinshipModel, ImageProcessor

class KinFaceWDataset(Dataset):
    """Dataset class for KinFaceW-I and KinFaceW-II"""
    def __init__(self, base_path, relation, fold, split='test', dataset_type='KinFaceW-I'):
        try:
            self.base_path = base_path
            self.relation = relation
            self.dataset_type = dataset_type
            
            # Map relation names to folder names
            self.relation_map = {
                'fs': 'father-son',
                'fd': 'father-dau',
                'ms': 'mother-son',
                'md': 'mother-dau'
            }
            
            # Load meta data
            meta_path = os.path.join(base_path, dataset_type, 'meta_data', f'{relation}_pairs.mat')
            print(f"\nDEBUG: Loading meta data from: {meta_path}")
            
            meta_data = scipy.io.loadmat(meta_path)
            print(f"DEBUG: Meta data keys: {meta_data.keys()}")
            
            # Get pairs data - shape is (N, 4) where N is number of pairs
            # Each row contains [fold, label, img1_path, img2_path]
            pairs_data = meta_data['pairs']
            print(f"DEBUG: Pairs data shape: {pairs_data.shape}")
            
            # Extract information
            # Convert fold numbers from 2D array to 1D
            fold_info = np.array([item[0][0] for item in pairs_data[:, 0]])
            self.labels = np.array([item[0][0] for item in pairs_data[:, 1]])
            
            # Extract image paths
            img1_paths = [item[0] for item in pairs_data[:, 2]]
            img2_paths = [item[0] for item in pairs_data[:, 3]]
            
            print(f"DEBUG: Fold info shape: {fold_info.shape}")
            print(f"DEBUG: Labels shape: {self.labels.shape}")
            
            # Create mask for the specified fold
            if split == 'test':
                self.pairs_mask = (fold_info == fold)
            else:  # train
                self.pairs_mask = (fold_info != fold)
            
            # Filter data based on fold mask
            self.labels = self.labels[self.pairs_mask].astype(np.float32)
            self.pairs = list(zip(
                np.array(img1_paths)[self.pairs_mask],
                np.array(img2_paths)[self.pairs_mask]
            ))
            
            print(f"DEBUG: Number of pairs for {split} split: {len(self.pairs)}")
            if len(self.pairs) > 0:
                print(f"DEBUG: Sample pair: {self.pairs[0]}")
            
        except Exception as e:
            print(f"\nDEBUG: Error in initialization:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Error location: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            raise
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        try:
            img1_name, img2_name = self.pairs[idx]
            
            # The image names already include the .jpg extension
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
            
            img1 = ImageProcessor.process_face(img1_path)
            img2 = ImageProcessor.process_face(img2_path)
            
            if img1 is None or img2 is None:
                print(f"\nDEBUG: Image processing failed for idx {idx}")
                print(f"Image paths: \n{img1_path}\n{img2_path}")
                img1 = torch.zeros(3, 112, 112)
                img2 = torch.zeros(3, 112, 112)
            
            return {
                'img1': img1,
                'img2': img2,
                'label': torch.tensor(self.labels[idx], dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"\nDEBUG: Error in __getitem__:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Index: {idx}")
            raise


def evaluate_fold(model, dataloader, device):
    """Evaluate model on a single fold"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(img1, img2)
            probs = torch.sigmoid(outputs['kinship_score'])
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics

def evaluate_dataset(model, base_path, dataset_type, device, batch_size=32):
    """Evaluate model on entire dataset (all relations, all folds)"""
    relations = ['fs', 'fd', 'ms', 'md']
    n_folds = 5
    
    all_results = {relation: [] for relation in relations}
    
    for relation in relations:
        print(f"\nEvaluating {relation} relation...")
        for fold in range(1, n_folds + 1):
            print(f"\nFold {fold}")
            
            # Create test dataset for this fold
            test_dataset = KinFaceWDataset(
                base_path=base_path,
                relation=relation,
                fold=fold,
                split='test',
                dataset_type=dataset_type
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Evaluate fold
            metrics = evaluate_fold(model, test_loader, device)
            metrics['fold'] = fold
            all_results[relation].append(metrics)
            
            print(f"Fold {fold} metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
    
    return all_results

def save_results(results, dataset_type, output_dir):
    """Save evaluation results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{dataset_type}_results_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save detailed results as JSON
    with open(os.path.join(output_path, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create summary DataFrame
    summary_data = []
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for relation in results:
        relation_metrics = {metric: [] for metric in metrics}
        for fold in results[relation]:
            for metric in metrics:
                relation_metrics[metric].append(fold[metric])
        
        summary_row = {
            'relation': relation,
            **{f'{metric}_mean': np.mean(relation_metrics[metric]) for metric in metrics},
            **{f'{metric}_std': np.std(relation_metrics[metric]) for metric in metrics}
        }
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, "summary_results.csv"), index=False)
    
    # Print summary
    print(f"\nResults saved to: {output_path}")
    print("\nSummary of results:")
    print(summary_df.to_string())


def inspect_mat_file(file_path):
    """Helper function to inspect the structure of a .mat file"""
    try:
        print(f"\nInspecting .mat file: {file_path}")
        data = scipy.io.loadmat(file_path)
        
        print("\nTop level keys:")
        for key in data.keys():
            if not key.startswith('__'):  # Skip metadata keys
                print(f"\nKey: {key}")
                value = data[key]
                print(f"Type: {type(value)}")
                print(f"Shape: {value.shape}")
                
                if isinstance(value, np.ndarray):
                    if value.dtype.names:  # Structured array
                        print("Fields:", value.dtype.names)
                        for field in value.dtype.names:
                            print(f"\nField '{field}':")
                            field_value = value[field]
                            print(f"Type: {type(field_value)}")
                            print(f"Shape: {field_value.shape}")
                            print(f"Sample: {field_value[:2]}")
                    else:
                        print(f"Sample: {value[:2]}")
        
    except Exception as e:
        print(f"Error inspecting file: {str(e)}")

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup
    config = KinshipConfig()
    model = KinshipModel(config).to(device)
    
    # Load your trained model
    model_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model/model/best_kin_nonkin_model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Base path for datasets
    base_path = "/cephyr/users/mehdiyev/Alvis/kinship_project/data"  
    output_dir = "/cephyr/users/mehdiyev/Alvis/kinship_project/src/evaluations/results_kinfacew"  # Update this path
    
    # Inspect structure of one .mat file first
    test_file = os.path.join(base_path, "KinFaceW-I", "meta_data", "fs_pairs.mat")
    inspect_mat_file(test_file)
    
    # Evaluate on KinFaceW-I
    print("\nEvaluating on KinFaceW-I dataset...")
    results_I = evaluate_dataset(model, base_path, "KinFaceW-I", device)
    save_results(results_I, "KinFaceW-I", output_dir)
    
    # Evaluate on KinFaceW-II
    print("\nEvaluating on KinFaceW-II dataset...")
    results_II = evaluate_dataset(model, base_path, "KinFaceW-II", device)
    save_results(results_II, "KinFaceW-II", output_dir)

if __name__ == "__main__":
    main()