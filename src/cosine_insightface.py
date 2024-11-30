import os
import torch
import tensorflow as tf 
import numpy as np
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import cv2
import onnxruntime as ort

# Configure CUDA environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
ort.set_default_logger_severity(3)  # Reduce logging noise

def process_image(image_path, app):
    try:
        # Read image using cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get face embedding
        faces = app.get(img)
        if not faces:
            print(f"No face detected in {image_path}")
            return None
            
        # Get embedding and ensure it's the correct shape
        embedding = faces[0].embedding
        if embedding is None or len(embedding.shape) != 1:
            print(f"Invalid embedding shape for {image_path}")
            return None
            
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def analyze_kinship_with_insightface(csv_path, output_dir, model_name="buffalo_l", low_kin_threshold=0.5, high_nonkin_threshold=0.6):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize InsightFace with CUDA settings
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider'
    ]
    
    # Initialize FaceAnalysis with specific CUDA settings
    app = FaceAnalysis(
        name=model_name,
        allowed_modules=['detection', 'recognition'],
        providers=providers
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    data = pd.read_csv(csv_path)
    kin_similarities = []
    nonkin_similarities = []
    low_kin_pairs = []
    high_nonkin_pairs = []
    
    print(f"Processing {len(data)} triplets using InsightFace {model_name} model on GPU...")
    
    # Process in batches for better GPU utilization
    batch_size = 32
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            anchor_path = row['Anchor']
            positive_path = row['Positive']
            negative_path = row['Negative']
            
            # Process images
            anchor_embed = process_image(anchor_path, app)
            if anchor_embed is None:
                continue
                
            positive_embed = process_image(positive_path, app)
            if positive_embed is None:
                continue
                
            negative_embed = process_image(negative_path, app)
            if negative_embed is None:
                continue
                
            # Verify embedding dimensions
            if anchor_embed.shape != positive_embed.shape or anchor_embed.shape != negative_embed.shape:
                print(f"Embedding dimension mismatch: {anchor_embed.shape}, {positive_embed.shape}, {negative_embed.shape}")
                continue
                
            # Calculate cosine similarities
            kin_sim = np.dot(anchor_embed, positive_embed)
            nonkin_sim = np.dot(anchor_embed, negative_embed)
            
            kin_similarities.append(kin_sim)
            nonkin_similarities.append(nonkin_sim)
            
            if kin_sim < low_kin_threshold:
                low_kin_pairs.append({
                    'anchor_path': anchor_path,
                    'positive_path': positive_path,
                    'similarity': kin_sim
                })
                
            if nonkin_sim > high_nonkin_threshold:
                high_nonkin_pairs.append({
                    'anchor_path': anchor_path,
                    'negative_path': negative_path,
                    'similarity': nonkin_sim
                })
    
    if not kin_similarities:
        print("No valid pairs were processed!")
        return None, None, None, None
    
    # Save results
    pd.DataFrame(low_kin_pairs).to_csv(os.path.join(output_dir, 'low_similarity_kin_pairs.csv'), index=False)
    pd.DataFrame(high_nonkin_pairs).to_csv(os.path.join(output_dir, 'high_similarity_nonkin_pairs.csv'), index=False)
    np.save(os.path.join(output_dir, 'kin_similarities.npy'), kin_similarities)
    np.save(os.path.join(output_dir, 'nonkin_similarities.npy'), nonkin_similarities)
    
    # Threshold metrics
    kin_similarities = np.array(kin_similarities)
    nonkin_similarities = np.array(nonkin_similarities)
    labels = np.concatenate([np.ones_like(kin_similarities), np.zeros_like(nonkin_similarities)])
    scores = np.concatenate([kin_similarities, nonkin_similarities])
    
    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    print(f"AUC: {auc:.4f}")
    
    best_threshold_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_idx]
    predictions = scores >= best_threshold
    print(classification_report(labels, predictions))
    
    return kin_similarities, nonkin_similarities, auc, best_threshold

if __name__ == "__main__":
    # Set environment variables for CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    
    # Set ONNX Runtime environment variables
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'
    
    csv_path = '../data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv'
    output_dir = 'evaluations/cosine_insightface_buffalo_l'
    
    try:
        kin_sims, nonkin_sims, auc, best_threshold = analyze_kinship_with_insightface(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="buffalo_l"
        )
        if auc is not None:
            print(f"Best Threshold: {best_threshold}")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")