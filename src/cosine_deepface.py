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
    embeddings_file = os.path.join(embeddings_dir, 'facenet512_embeddings.pkl')
    with open(embeddings_file, 'rb') as f:
        return pickle.load(f)

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
    
    kin_similarities = np.array(kin_similarities)
    nonkin_similarities = np.array(nonkin_similarities)
    
    # Calculate metrics across different thresholds
    threshold_metrics = []
    best_accuracy = 0
    best_metrics = None
    
    print("\nCalculating threshold metrics...")
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
    
    # Save results
    print("\nSaving results...")
    
    # Save similarities
    np.save(os.path.join(output_dir, 'kin_similarities.npy'), kin_similarities)
    np.save(os.path.join(output_dir, 'nonkin_similarities.npy'), nonkin_similarities)
    
    # Save metrics
    pd.DataFrame(threshold_metrics).to_csv(
        os.path.join(output_dir, 'threshold_metrics.csv'), index=False)
    
    # Save failed triplets
    with open(os.path.join(output_dir, 'failed_triplets.txt'), 'w') as f:
        for idx, error in failed_triplets:
            f.write(f"Triplet {idx}: {error}\n")
    
    # Generate and save plots
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
    
    # Calculate final metrics
    labels = np.concatenate([np.ones_like(kin_similarities), np.zeros_like(nonkin_similarities)])
    scores = np.concatenate([kin_similarities, nonkin_similarities])
    
    # ROC curve
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
    
    # Print final results
    print("\nFinal Results:")
    print(f"Processed triplets: {len(kin_similarities)}")
    print(f"Failed triplets: {len(failed_triplets)}")
    print(f"AUC: {auc:.4f}")
    print(f"Best threshold: {best_metrics['threshold']:.4f}")
    print(f"Best accuracy: {best_metrics['accuracy']:.4f}")
    
    # Detailed classification report
    predictions = scores >= best_metrics['threshold']
    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions))
    
    return kin_similarities, nonkin_similarities, best_metrics

if __name__ == "__main__":
    csv_path = '../data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv'
    embeddings_dir = 'facenet512_embeddings'
    output_dir = 'kinship_analysis_results'
    
    try:
        kin_sims, nonkin_sims, best_metrics = analyze_kinship_from_embeddings(
            csv_path=csv_path,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            sample_size=None  # Set to None to process all triplets
        )
        print("\nAnalysis completed successfully.")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()