import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
ort.set_default_logger_severity(3)

def process_image(image_path, recognition_model, verbose=False):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}")
            return None
            
        if verbose:
            print(f"\nProcessing: {image_path}")
            print(f"Original shape: {img.shape}")
            print(f"Value range: [{img.min()}, {img.max()}]")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img.shape[:2] != (112, 112):
            if verbose:
                print(f"Resizing from {img.shape[:2]} to (112, 112)")
            img = cv2.resize(img, (112, 112))        
        
        # Get embedding
        embedding = recognition_model.get_feat(img)
        if embedding is None:
            print(f"Got None embedding for {image_path}")
            return None
            
        embedding = embedding.flatten()
        
        if verbose:
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding range: [{embedding.min():.2f}, {embedding.max():.2f}]")
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            print(f"Warning: Very small norm {norm} for {image_path}")
            return None
            
        embedding = embedding / norm
        
        if verbose:
            final_norm = np.linalg.norm(embedding)
            print(f"Final embedding norm: {final_norm:.6f}")
        
        return embedding
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_processing(recognition_model, data, n_samples=5):
    print("\nValidating image processing with sample images:")
    valid_samples = 0
    
    for i in range(min(n_samples, len(data))):
        sample_row = data.iloc[i]
        print(f"\nSample {i+1}:")
        print(f"Anchor: {sample_row['Anchor']}")
        
        # Process and validate anchor
        anchor_embed = process_image(sample_row['Anchor'], recognition_model, verbose=True)
        if anchor_embed is None:
            print("✗ Anchor processing failed")
            continue
        print("✓ Anchor processed successfully")
        
        # Process and validate positive
        positive_embed = process_image(sample_row['Positive'], recognition_model, verbose=True)
        if positive_embed is None:
            print("✗ Positive processing failed")
            continue
        print("✓ Positive processed successfully")
        
        # Calculate and check kin similarity
        kin_sim = np.dot(anchor_embed, positive_embed)
        print(f"Kin similarity: {kin_sim:.4f}")
        
        # Process and validate negative
        negative_embed = process_image(sample_row['Negative'], recognition_model, verbose=True)
        if negative_embed is None:
            print("✗ Negative processing failed")
            continue
        print("✓ Negative processed successfully")
        
        # Calculate and check non-kin similarity
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
        
        # Verify all files were saved
        expected_files = [
            'kin_similarities.npy',
            'nonkin_similarities.npy',
            'similarities.csv',
            'threshold_metrics.csv',
            'statistics.txt',
            'best_metrics.txt',
            'similarity_distributions.png',
            'roc_curve.png'
        ]
        
        missing_files = []
        for file in expected_files:
            file_path = os.path.join(output_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"\nWarning: The following files were not saved: {missing_files}")
        else:
            print("\nAll result files were saved successfully.")
            print(f"Results directory: {output_dir}")
            # List all saved files with sizes
            print("\nSaved files:")
            for file in expected_files:
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"  - {file}: {size/1024:.1f} KB")
                
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()

def calculate_threshold_metrics(kin_similarities, nonkin_similarities):
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

def analyze_kinship_with_insightface(csv_path, output_dir, model_name="buffalo_l", sample_size=None):
    # Create output directory with parents if needed
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    
    app = FaceAnalysis(
        name=model_name,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0)
    recognition_model = app.models.get('recognition')
    print(f"Using model: {recognition_model}")
    
    # Print model information
    print(f"\nRecognition model input shape: {recognition_model.input_shape}")
    print(f"Recognition model output shape: {recognition_model.output_shape}")
    
    # Load and validate CSV
    data = pd.read_csv(csv_path)
    print("\nCSV columns:", data.columns)
    print("First row:", data.iloc[0])
    print("Total rows:", len(data))
    
    if sample_size is not None:
        data = data.sample(n=min(sample_size, len(data)), random_state=42)
        print(f"Using random sample of {len(data)} rows")
    
    # Validate processing with sample images
    if not validate_processing(recognition_model, data):
        print("\nValidation failed! Please check the sample processing results above.")
        return None, None
    
    # user_input = input("\nContinue with full processing? (y/n): ")
    # if user_input.lower() != 'y':
    #     print("Processing cancelled")
    #     return None, None
    
    # Process all images
    kin_similarities = []
    nonkin_similarities = []
    processed_pairs = 0
    failed_pairs = 0
    
    print(f"\nProcessing {len(data)} triplets using InsightFace {model_name} model...")
    
    batch_size = 256
    for i in tqdm(range(0, len(data), batch_size)):
        # at each 5% print the progress
        if i % (len(data) // 20) == 0:
            print(f"Processing {i}/{len(data)} triplets")
        
        batch = data.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            # Process images
            anchor_embed = process_image(row['Anchor'], recognition_model)
            if anchor_embed is None:
                failed_pairs += 1
                continue
            
            positive_embed = process_image(row['Positive'], recognition_model)
            if positive_embed is None:
                failed_pairs += 1
                continue
            
            negative_embed = process_image(row['Negative'], recognition_model)
            if negative_embed is None:
                failed_pairs += 1
                continue
            
            # Calculate similarities
            kin_sim = np.dot(anchor_embed, positive_embed)
            nonkin_sim = np.dot(anchor_embed, negative_embed)
            
            kin_similarities.append(kin_sim)
            nonkin_similarities.append(nonkin_sim)
            processed_pairs += 1
    
    if not kin_similarities:
        print("No valid pairs were processed!")
        return None, None
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_pairs} pairs")
    print(f"Failed to process: {failed_pairs} pairs")
    
    # Convert to numpy arrays
    kin_similarities = np.array(kin_similarities)
    nonkin_similarities = np.array(nonkin_similarities)
    
    # Calculate metrics
    threshold_metrics, best_metrics = calculate_threshold_metrics(kin_similarities, nonkin_similarities)
    
    # Save all results
    save_all_results(output_dir, kin_similarities, nonkin_similarities, threshold_metrics, best_metrics)
    
    # Print some results
    labels = np.concatenate([np.ones_like(kin_similarities), np.zeros_like(nonkin_similarities)])
    
    # Calculate final metrics for display
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
    output_dir = 'evaluations/cosine_insightface_buffalo_l'
    
    try:
        # Run the analysis
        kin_sims, nonkin_sims = analyze_kinship_with_insightface(
            csv_path=csv_path,
            output_dir=output_dir,
            sample_size=None
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