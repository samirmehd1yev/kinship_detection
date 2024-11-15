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
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import KinshipConfig and KinshipModel from kin_nonkin_training_v5.py
# Assume kin_nonkin_training_v5.py is in the same directory or adjust the import path accordingly
from kin_nonkin_training_v5 import KinshipConfig, KinshipModel


# Custom image processing functions
class ImageProcessor:
    def __init__(self, config):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            # Mean: [0.49684819 0.38204407 0.33240583], Std: [0.30901513 0.26008384 0.24445895]
            transforms.Normalize(mean=[0.49684819, 0.38204407, 0.33240583], std=[0.30901513, 0.26008384, 0.24445895])
        ])

    def process_face(self, img_path):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None

class TripletDataset(Dataset):
    def __init__(self, csv_path, processor, sample_size=None):
        self.df = pd.read_csv(csv_path)
        if sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42)
        self.processor = processor
        self.retry_count = 0  # Add retry counter
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.retry_count > 10:  # Prevent infinite recursion
            self.retry_count = 0
            raise RuntimeError("Failed to load valid images after 10 retries")
            
        row = self.df.iloc[idx]
        anchor_path = row['Anchor']
        positive_path = row['Positive']
        negative_path = row['Negative']
        
        anchor = self.processor.process_face(anchor_path)
        positive = self.processor.process_face(positive_path)
        negative = self.processor.process_face(negative_path)

        if anchor is None or positive is None or negative is None:
            self.retry_count += 1
            return self.__getitem__((idx + 1) % len(self))
            
        self.retry_count = 0  # Reset counter on successful load
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_path': anchor_path,
            'positive_path': positive_path,
            'negative_path': negative_path
        }
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
    """
    Analyze similarities between kin and non-kin pairs using batched processing.
    
    Args:
        csv_path (str): Path to CSV containing triplets
        model_path (str): Path to trained model checkpoint
        batch_size (int): Batch size for processing
        sample_size (int, optional): Number of samples to process. If None, process all
        
    Returns:
        tuple: Arrays of kin and non-kin similarities
    """
    # Load model
    config = KinshipConfig()
    model = KinshipModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create ImageProcessor
    processor = ImageProcessor(config)

    # Create dataset and dataloader
    dataset = TripletDataset(csv_path, processor, sample_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    # Lists to store similarities
    kin_similarities = []
    nonkin_similarities = []
    
    # Initialize lists to store pairs
    high_nonkin_pairs = []
    low_kin_pairs = []
    
    print(f"Processing {len(dataset)} triplets in batches of {batch_size}")
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Processing Batches'):
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
                
                # Get paths from batch
                anchor_paths = batch['anchor_path']
                positive_paths = batch['positive_path']
                negative_paths = batch['negative_path']
                
                # Define similarity thresholds
                high_sim_threshold = 0.6  # Adjust as needed
                low_sim_threshold = 0.5   # Adjust as needed
                
                # Store interesting pairs
                for idx in range(len(kin_sim)):
                    sim = kin_sim[idx].item()
                    if sim < low_sim_threshold:
                        low_kin_pairs.append({
                            'anchor_path': anchor_paths[idx],
                            'positive_path': positive_paths[idx],
                            'similarity': sim
                        })
                
                for idx in range(len(nonkin_sim)):
                    sim = nonkin_sim[idx].item()
                    if sim > high_sim_threshold:
                        high_nonkin_pairs.append({
                            'anchor_path': anchor_paths[idx],
                            'negative_path': negative_paths[idx],
                            'similarity': sim
                        })
                
                # Store results
                kin_similarities.extend(kin_sim.cpu().numpy())
                nonkin_similarities.extend(nonkin_sim.cpu().numpy())

                # Clear GPU cache periodically
                if torch.cuda.is_available() and len(kin_similarities) % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        raise
    
    # Convert to numpy arrays
    kin_similarities = np.array(kin_similarities)
    nonkin_similarities = np.array(nonkin_similarities)
    
    # Verify equal numbers and print distribution info
    assert len(kin_similarities) == len(nonkin_similarities), \
        f"Unequal pairs: kin={len(kin_similarities)}, nonkin={len(nonkin_similarities)}"
    
    print("\nDistribution Analysis:")
    print(f"Total pairs: {len(kin_similarities)} kin, {len(nonkin_similarities)} nonkin")
    print(f"Kin similarities range: [{kin_similarities.min():.3f}, {kin_similarities.max():.3f}]")
    print(f"Nonkin similarities range: [{nonkin_similarities.min():.3f}, {nonkin_similarities.max():.3f}]")
    print(f"Kin similarities < 0: {np.sum(kin_similarities < 0)}")
    print(f"Nonkin similarities < 0: {np.sum(nonkin_similarities < 0)}")
    
    # Calculate Bhattacharyya coefficient
    bc_kin_nonkin = calculate_bhattacharyya_coefficient(kin_similarities, nonkin_similarities)
    
    # Create output directory
    output_dir = '/cephyr/users/mehdiyev/Alvis/kinship_project/src/evaluations/cosine2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistics to file
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write("\nKin pairs statistics:\n")
        f.write(f"Count: {len(kin_similarities)}\n")
        f.write(f"Mean: {np.mean(kin_similarities):.4f}\n")
        f.write(f"Std: {np.std(kin_similarities):.4f}\n")
        f.write(f"Min: {np.min(kin_similarities):.4f}\n")
        f.write(f"Max: {np.max(kin_similarities):.4f}\n")
        f.write(f"Values < 0: {np.sum(kin_similarities < 0)}\n")
        
        f.write("\nNon-kin pairs statistics:\n")
        f.write(f"Count: {len(nonkin_similarities)}\n")
        f.write(f"Mean: {np.mean(nonkin_similarities):.4f}\n")
        f.write(f"Std: {np.std(nonkin_similarities):.4f}\n")
        f.write(f"Min: {np.min(nonkin_similarities):.4f}\n")
        f.write(f"Max: {np.max(nonkin_similarities):.4f}\n")
        f.write(f"Values < 0: {np.sum(nonkin_similarities < 0)}\n")
        
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
    thresholds = np.linspace(-1, 1, 100)  # Changed to include full cosine similarity range
    tpr, fpr = [], []
    
    for threshold in thresholds:
        tp = np.sum(kin_similarities >= threshold)
        fn = np.sum(kin_similarities < threshold)
        fp = np.sum(nonkin_similarities >= threshold)
        tn = np.sum(nonkin_similarities < threshold)
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
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
    
    # Calculate threshold metrics with full range
    threshold_metrics = []
    for threshold in np.arange(-1, 1.01, 0.05):  # Changed to include full range
        tp = np.sum((kin_similarities >= threshold))
        tn = np.sum((nonkin_similarities < threshold))
        fp = np.sum((nonkin_similarities >= threshold))
        fn = np.sum((kin_similarities < threshold))
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
        
        # Print metrics at threshold 0.0 for verification
        if abs(threshold) < 1e-6:
            print("\nMetrics at threshold 0.0:")
            print(f"True Positives: {tp}")
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
    
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
    
    # Save high similarity non-kin pairs and low similarity kin pairs
    high_nonkin_df = pd.DataFrame(high_nonkin_pairs)
    high_nonkin_df.to_csv(os.path.join(output_dir, 'high_similarity_nonkin_pairs.csv'), index=False)
    
    low_kin_df = pd.DataFrame(low_kin_pairs)
    low_kin_df.to_csv(os.path.join(output_dir, 'low_similarity_kin_pairs.csv'), index=False)
    
    return kin_similarities, nonkin_similarities

if __name__ == "__main__":
    # Paths
    csv_path = '/cephyr/users/mehdiyev/Alvis/kinship_project/data/processed/fiw/train/filtered_triplets_with_labels.csv'
    model_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model_v5/model/best_kin_nonkin_model.pth'
    
    # Analyze similarities with batches
    kin_sims, nonkin_sims = analyze_similarities_batch(
        csv_path=csv_path,
        model_path=model_path,
        # sample_size=1000,
        batch_size=128
    )