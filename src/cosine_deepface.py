import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

def process_image(image_path, model_name):
    try:
        # Extract embedding for the given image
        embedding = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
        return np.array(embedding[0]['embedding'])
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def analyze_kinship_with_deepface(csv_path, output_dir, model_name="VGG-Face", low_kin_threshold=0.5, high_nonkin_threshold=0.6):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_path)
    
    kin_similarities = []
    nonkin_similarities = []
    low_kin_pairs = []
    high_nonkin_pairs = []
    
    print(f"Processing {len(data)} triplets using {model_name} model...")
    
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        anchor_path = row['Anchor']
        positive_path = row['Positive']
        negative_path = row['Negative']
        
        anchor_embed = process_image(anchor_path, model_name)
        positive_embed = process_image(positive_path, model_name)
        negative_embed = process_image(negative_path, model_name)
        
        if anchor_embed is None or positive_embed is None or negative_embed is None:
            continue
        
        kin_sim = np.dot(anchor_embed, positive_embed) / (np.linalg.norm(anchor_embed) * np.linalg.norm(positive_embed))
        nonkin_sim = np.dot(anchor_embed, negative_embed) / (np.linalg.norm(anchor_embed) * np.linalg.norm(negative_embed))
        
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
    csv_path = '/cephyr/users/mehdiyev/Alvis/kinship_project/data/processed/fiw/train/filtered_triplets_with_labels.csv'
    output_dir = '/cephyr/users/mehdiyev/Alvis/kinship_project/src/evaluations/deepface'
    
    kin_sims, nonkin_sims, auc, best_threshold = analyze_kinship_with_deepface(
        csv_path=csv_path,
        output_dir=output_dir,
        model_name="ArcFace"
    )
    print(f"Best Threshold: {best_threshold}")
