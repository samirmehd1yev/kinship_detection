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
from kin_nonkin_training_v5 import KinshipConfig, KinshipModel, ImageProcessor

class KinFaceWDataset(Dataset):
    """Dataset class for KinFaceW-I and KinFaceW-II"""
    def __init__(self, base_path, relation, split='test', dataset_type='KinFaceW-I', config=None):
        try:
            self.base_path = base_path
            self.relation = relation
            self.dataset_type = dataset_type
            self.processor = ImageProcessor(config)
            
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
            
            # Get pairs data
            pairs_data = meta_data['pairs']
            print(f"DEBUG: Pairs data shape: {pairs_data.shape}")
            
            # Extract labels and image paths
            self.labels = np.array([item[0][0] for item in pairs_data[:, 1]]).astype(np.float32)
            img1_paths = [item[0] for item in pairs_data[:, 2]]
            img2_paths = [item[0] for item in pairs_data[:, 3]]
            
            # Create list of pairs
            self.pairs = list(zip(img1_paths, img2_paths))
            
            print(f"DEBUG: Total number of pairs: {len(self.pairs)}")
            if len(self.pairs) > 0:
                print(f"DEBUG: Sample pair: {self.pairs[0]}")
                
        except Exception as e:
            print(f"\nDEBUG: Error in initialization:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            img1_name, img2_name = self.pairs[idx]
            
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
            
            img1 = self.processor.process_face(img1_path)
            img2 = self.processor.process_face(img2_path)
            
            if img1 is None or img2 is None:
                print(f"\nDEBUG: Image processing failed for idx {idx}")
                print(f"Image paths: \n{img1_path}\n{img2_path}")
                img1 = torch.zeros(3, self.processor.config.input_size, self.processor.config.input_size)
                img2 = torch.zeros(3, self.processor.config.input_size, self.processor.config.input_size)
            
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

def evaluate_dataset(model, base_path, dataset_type, device, config, batch_size=32):
    """Evaluate model on entire dataset for kin/non-kin classification"""
    relations = ['fs', 'fd', 'ms', 'md']
    
    all_results = {}
    
    for relation in relations:
        print(f"\nEvaluating relation: {relation}")
        
        # Create dataset for the relation
        dataset = KinFaceWDataset(
            base_path=base_path,
            relation=relation,
            split='test',
            dataset_type=dataset_type,
            config=config
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Evaluate on the dataset
        metrics = evaluate_fold(model, dataloader, device)
        all_results[relation] = metrics
        
        print(f"Metrics for relation {relation}:")
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
        relation_metrics = results[relation]
        
        summary_row = {
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
    config = KinshipConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup
    model = KinshipModel(config).to(device)
    
    # Load your trained model
    model_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model_v5/model/best_kin_nonkin_model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Base path for datasets
    base_path = "/cephyr/users/mehdiyev/Alvis/kinship_project/data"
    output_dir = "/cephyr/users/mehdiyev/Alvis/kinship_project/src/evaluations/results_kinfacew"
    
    # Evaluate on KinFaceW-I
    print("\nEvaluating on KinFaceW-I dataset...")
    results_I = evaluate_dataset(model, base_path, "KinFaceW-I", device, config)
    save_results(results_I, "KinFaceW-I", output_dir)
    
    # Evaluate on KinFaceW-II
    print("\nEvaluating on KinFaceW-II dataset...")
    results_II = evaluate_dataset(model, base_path, "KinFaceW-II", device, config)
    save_results(results_II, "KinFaceW-II", output_dir)

if __name__ == "__main__":
    main()