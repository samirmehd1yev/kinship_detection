import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_embeddings(embeddings_dir):
    """Load the saved embeddings dictionary"""
    embeddings_file = os.path.join(embeddings_dir, 'arcface_embeddings.pkl')
    with open(embeddings_file, 'rb') as f:
        return pickle.load(f)

def validate_embeddings(embeddings_dict, data, n_samples=5):
    """Validate embeddings with sample images"""
    print("\nValidating embeddings with sample images:")
    valid_samples = 0
    
    for i in range(min(n_samples, len(data))):
        sample_row = data.iloc[i]
        print(f"\nSample {i+1}:")
        
        # Get image keys (basenames)
        anchor_key = os.path.basename(sample_row['Anchor'])
        positive_key = os.path.basename(sample_row['Positive'])
        negative_key = os.path.basename(sample_row['Negative'])
        
        print(f"Anchor: {anchor_key}")
        
        # Validate anchor
        if anchor_key not in embeddings_dict:
            print("✗ Anchor embedding not found")
            continue
        print("✓ Anchor embedding found")
        
        # Validate positive
        if positive_key not in embeddings_dict:
            print("✗ Positive embedding not found")
            continue
        print("✓ Positive embedding found")
        
        # Calculate and check kin similarity
        anchor_embed = embeddings_dict[anchor_key]['embedding']
        positive_embed = embeddings_dict[positive_key]['embedding']
        kin_sim = np.dot(anchor_embed, positive_embed)
        print(f"Kin similarity: {kin_sim:.4f}")
        
        # Validate negative
        if negative_key not in embeddings_dict:
            print("✗ Negative embedding not found")
            continue
        print("✓ Negative embedding found")
        
        # Calculate and check non-kin similarity
        negative_embed = embeddings_dict[negative_key]['embedding']
        nonkin_sim = np.dot(anchor_embed, negative_embed)
        print(f"Non-kin similarity: {nonkin_sim:.4f}")
        
        valid_samples += 1
        print("-" * 50)
    
    print(f"\nValidation complete: {valid_samples}/{n_samples} samples processed successfully")
    return valid_samples > 0

def save_all_results(output_dir, kin_similarities, nonkin_similarities, threshold_metrics, best_metrics):
    """Helper function to ensure all results are saved"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nSaving results...")
        
        # 1. Save raw similarities
        np.save(os.path.join(output_dir, 'kin_similarities.npy'), kin_similarities)
        np.save(os.path.join(output_dir, 'nonkin_similarities.npy'), nonkin_similarities)
        
        # 2. Save CSV files
        pd.DataFrame({
            'kin_similarities': kin_similarities,
            'nonkin_similarities': nonkin_similarities
        }).to_csv(os.path.join(output_dir, 'similarities.csv'), index=False)
        
        pd.DataFrame(threshold_metrics).to_csv(
            os.path.join(output_dir, 'threshold_metrics.csv'), index=False)
        
        # 3. Save statistics
        with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
            f.write("Kin pairs statistics:\n")
            f.write(f"Count: {len(kin_similarities)}\n")
            f.write(f"Mean: {np.mean(kin_similarities):.4f}\n")
            f.write(f"Std: {np.std(kin_similarities):.4f}\n")
            f.write(f"Min: {np.min(kin_similarities):.4f}\n")
            f.write(f"Max: {np.max(kin_similarities):.4f}\n\n")
            
            f.write("Non-kin pairs statistics:\n")
            f.write(f"Count: {len(nonkin_similarities)}\n")
            f.write(f"Mean: {np.mean(nonkin_similarities):.4f}\n")
            f.write(f"Std: {np.std(nonkin_similarities):.4f}\n")
            f.write(f"Min: {np.min(nonkin_similarities):.4f}\n")
            f.write(f"Max: {np.max(nonkin_similarities):.4f}\n")
        
        # 4. Save best metrics
        with open(os.path.join(output_dir, 'best_metrics.txt'), 'w') as f:
            f.write("Best performing threshold metrics:\n")
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # 5. Save plots
        # Distribution plot
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=kin_similarities, label='Kin Pairs', color='blue', fill=True, alpha=0.3)
        sns.kdeplot(data=nonkin_similarities, label='Non-kin Pairs', color='red', fill=True, alpha=0.3)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Distribution of Cosine Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve plot
        labels = np.concatenate([np.ones_like(kin_similarities), np.zeros_like(nonkin_similarities)])
        scores = np.concatenate([kin_similarities, nonkin_similarities])
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True, alpha=0.3)
        plt.text(0.6, 0.2, f'AUC = {auc:.3f}', fontsize=12)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nAll result files were saved successfully.")
        print(f"Results directory: {output_dir}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()

def calculate_threshold_metrics(kin_similarities, nonkin_similarities):
    """Calculate metrics for different thresholds"""
    threshold_metrics = []
    best_accuracy = 0
    best_metrics = None
    
    for threshold in np.arange(-1, 1.01, 0.05):
        tp = np.sum((kin_similarities >= threshold))
        tn = np.sum((nonkin_similarities < threshold))
        fp = np.sum((nonkin_similarities >= threshold))
        fn = np.sum((kin_similarities < threshold))
        
        total = len(kin_similarities) + len(nonkin_similarities)
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        threshold_metrics.append(metrics)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_metrics = metrics.copy()
    
    return threshold_metrics, best_metrics

def analyze_kinship_from_embeddings(csv_path, embeddings_dir, output_dir, sample_size=None):
    """Analyze kinship relationships using pre-computed embeddings"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embeddings
    print("\nLoading pre-computed embeddings...")
    embeddings_dict = load_embeddings(embeddings_dir)
    print(f"Loaded {len(embeddings_dict)} embeddings")
    
    # Load CSV
    data = pd.read_csv(csv_path)
    if sample_size:
        data = data.sample(n=min(sample_size, len(data)), random_state=42)
    print(f"\nAnalyzing {len(data)} triplets")
    
    # # Validate embeddings
    # if not validate_embeddings(embeddings_dict, data):
    #     print("\nValidation failed! Please check the embeddings.")
    #     return None, None
    
    # Process triplets
    kin_similarities = []
    nonkin_similarities = []
    failed_triplets = []
    
    print("\nCalculating similarities...")
    for idx, row in tqdm(data.iterrows()):
        try:
            # Get image keys (basenames)
            anchor_key = os.path.basename(row['Anchor'])
            positive_key = os.path.basename(row['Positive'])
            negative_key = os.path.basename(row['Negative'])
            
            # Get embeddings
            anchor_embed = embeddings_dict[anchor_key]['embedding']
            positive_embed = embeddings_dict[positive_key]['embedding']
            negative_embed = embeddings_dict[negative_key]['embedding']
            
            # Calculate similarities
            kin_sim = np.dot(anchor_embed, positive_embed)
            nonkin_sim = np.dot(anchor_embed, negative_embed)
            
            kin_similarities.append(kin_sim)
            nonkin_similarities.append(nonkin_sim)
            
        except KeyError as e:
            failed_triplets.append((idx, str(e)))
            continue
    
    if not kin_similarities:
        print("No valid pairs were processed!")
        return None, None
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {len(kin_similarities)} pairs")
    print(f"Failed to process: {len(failed_triplets)} pairs")
    
    # Convert to numpy arrays
    kin_similarities = np.array(kin_similarities)
    nonkin_similarities = np.array(nonkin_similarities)
    
    # Calculate metrics
    threshold_metrics, best_metrics = calculate_threshold_metrics(kin_similarities, nonkin_similarities)
    
    # Save all results
    save_all_results(output_dir, kin_similarities, nonkin_similarities, threshold_metrics, best_metrics)
    
    # Calculate final metrics for display
    labels = np.concatenate([np.ones_like(kin_similarities), np.zeros_like(nonkin_similarities)])
    scores = np.concatenate([kin_similarities, nonkin_similarities])
    auc = roc_auc_score(labels, scores)
    predictions = scores >= best_metrics['threshold']
    
    print("\nFinal Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Best threshold: {best_metrics['threshold']:.4f}")
    print(f"Best accuracy: {best_metrics['accuracy']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions))
    
    return kin_similarities, nonkin_similarities

if __name__ == "__main__":
    csv_path = '../data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv'
    embeddings_dir = 'arcface_embeddings'
    output_dir = 'evaluations/cosine_deepface_arcface'
    
    try:
        kin_sims, nonkin_sims = analyze_kinship_from_embeddings(
            csv_path=csv_path,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            sample_size=None  # Set to None to process all triplets
        )
        
        if kin_sims is None:
            print("\nAnalysis failed or was cancelled.")
        else:
            print("\nAnalysis completed successfully.")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
        
# (kinship_venv_tf) [mehdiyev@alvis1 src]$ python cosine_deepface.py 

# Loading pre-computed embeddings...
# Loaded 12613 embeddings

# Analyzing 181195 triplets

# Validating embeddings with sample images:

# Sample 1:
# Anchor: P00001_face0.jpg
# ✓ Anchor embedding found
# ✓ Positive embedding found
# Kin similarity: 0.1450
# ✓ Negative embedding found
# Non-kin similarity: 0.1489
# --------------------------------------------------

# Sample 2:
# Anchor: P00002_face0.jpg
# ✓ Anchor embedding found
# ✓ Positive embedding found
# Kin similarity: 0.1426
# ✓ Negative embedding found
# Non-kin similarity: -0.0780
# --------------------------------------------------

# Sample 3:
# Anchor: P00008_face4.jpg
# ✓ Anchor embedding found
# ✓ Positive embedding found
# Kin similarity: 0.2186
# ✓ Negative embedding found
# Non-kin similarity: -0.1857
# --------------------------------------------------

# Sample 4:
# Anchor: P00001_face0.jpg
# ✓ Anchor embedding found
# ✓ Positive embedding found
# Kin similarity: 0.1450
# ✓ Negative embedding found
# Non-kin similarity: -0.1171
# --------------------------------------------------

# Sample 5:
# Anchor: P00008_face4.jpg
# ✓ Anchor embedding found
# ✓ Positive embedding found
# Kin similarity: 0.2186
# ✓ Negative embedding found
# Non-kin similarity: -0.1666
# --------------------------------------------------

# Validation complete: 5/5 samples processed successfully

# Calculating similarities...
# 181195it [00:10, 17771.76it/s]

# Processing complete:
# Successfully processed: 181186 pairs
# Failed to process: 9 pairs

# Saving results...

# All result files were saved successfully.
# Results directory: evaluations/cosine_deepface_facenet512

# Final Results:
# AUC: 0.7766
# Best threshold: 0.2000
# Best accuracy: 0.7095

# Detailed Classification Report:
#               precision    recall  f1-score   support

#          0.0       0.68      0.79      0.73    181186
#          1.0       0.75      0.63      0.68    181186

#     accuracy                           0.71    362372
#    macro avg       0.72      0.71      0.71    362372
# weighted avg       0.72      0.71      0.71    362372


# Analysis completed successfully.
# (kinship_venv_tf) [mehdiyev@alvis1 src]$         