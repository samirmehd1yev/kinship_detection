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
from facenet_pytorch import InceptionResnetV1
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_image(image_path, target_size=299):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            # transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(img)
        
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

class TripletDataset(Dataset):
    def __init__(self, csv_path, sample_size=None):
        self.df = pd.read_csv(csv_path)
        if sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42)
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor_path = row['Anchor']
        positive_path = row['Positive']
        negative_path = row['Negative']
        
        anchor = load_and_preprocess_image(anchor_path)
        positive = load_and_preprocess_image(positive_path)
        negative = load_and_preprocess_image(negative_path)
        
        if anchor is None or positive is None or negative is None:
            return self.__getitem__(0)
                
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_path': anchor_path,
            'positive_path': positive_path,
            'negative_path': negative_path
        }

def get_inception_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.fc = nn.Identity()  # Remove classification head
    return model

def analyze_similarities_batch(csv_path, output_dir, batch_size=128, sample_size=None, low_kin_threshold=0.5, high_nonkin_threshold=0.6):
    os.makedirs(output_dir, exist_ok=True)
    
    model = get_inception_model().to(device)
    model.eval()
    
    try:
        dataset = TripletDataset(csv_path, sample_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    except Exception as e:
        raise Exception(f"Error creating dataset: {str(e)}")
    
        
    kin_similarities = []
    nonkin_similarities = []
    high_nonkin_pairs = [] 
    low_kin_pairs = []
    
    if sample_size:
        print(f"Processing {sample_size} triplets in batches of {batch_size}")
    else:
        print(f"Processing {len(dataset)} triplets in batches of {batch_size}")
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                anchor = batch['anchor'].to(device)
                positive = batch['positive'].to(device)
                negative = batch['negative'].to(device)
                
                anchor_feat = model(anchor)
                positive_feat = model(positive)
                negative_feat = model(negative)
                
                kin_sim = F.cosine_similarity(anchor_feat, positive_feat)
                nonkin_sim = F.cosine_similarity(anchor_feat, negative_feat)
                
                kin_similarities.extend(kin_sim.cpu().numpy())
                nonkin_similarities.extend(nonkin_sim.cpu().numpy())
                
                for idx in range(len(kin_sim)):
                    if kin_sim[idx].item() < low_kin_threshold:
                        low_kin_pairs.append({
                            'anchor_path': batch['anchor_path'][idx],
                            'positive_path': batch['positive_path'][idx],
                            'similarity': kin_sim[idx].item()
                        })
                    
                    if nonkin_sim[idx].item() > high_nonkin_threshold:
                        high_nonkin_pairs.append({
                            'anchor_path': batch['anchor_path'][idx],
                            'negative_path': batch['negative_path'][idx],
                            'similarity': nonkin_sim[idx].item()
                        })
                
                del anchor, positive, negative, anchor_feat, positive_feat, negative_feat
                torch.cuda.empty_cache()

    except Exception as e:
        raise Exception(f"Error during processing: {str(e)}")
       
    kin_similarities = np.array(kin_similarities)
    nonkin_similarities = np.array(nonkin_similarities)
   
    try:
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
        
        # Save best performing threshold metrics
        threshold_metrics = []
        best_accuracy = 0
        best_threshold = None
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
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_metrics = {
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
        
        # Save best metrics
        with open(os.path.join(output_dir, 'best_metrics.txt'), 'w') as f:
            f.write("Best performing threshold metrics:\n")
            f.write(f"Threshold: {best_metrics['threshold']:.4f}\n")
            f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {best_metrics['precision']:.4f}\n")
            f.write(f"Recall: {best_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {best_metrics['f1']:.4f}\n")
            f.write(f"True Positives: {best_metrics['tp']}\n")
            f.write(f"True Negatives: {best_metrics['tn']}\n")
            f.write(f"False Positives: {best_metrics['fp']}\n")
            f.write(f"False Negatives: {best_metrics['fn']}\n")
        
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
        
        thresholds = np.linspace(-1, 1, 100)
        tpr = []
        fpr = []
       
        for threshold in thresholds:
            tp = np.sum(kin_similarities >= threshold)
            fn = np.sum(kin_similarities < threshold)
            fp = np.sum(nonkin_similarities >= threshold)
            tn = np.sum(nonkin_similarities < threshold)
           
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
        
        auc = np.trapz(tpr, fpr)
       
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
        
        threshold_metrics = []
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
        
        pd.DataFrame(threshold_metrics).to_csv(
            os.path.join(output_dir, 'threshold_metrics.csv'), index=False)
       
        pd.DataFrame({
            'kin_similarities': kin_similarities,
            'nonkin_similarities': nonkin_similarities
        }).to_csv(os.path.join(output_dir, 'similarities.csv'), index=False)
       
        pd.DataFrame(high_nonkin_pairs).to_csv(
            os.path.join(output_dir, 'high_similarity_nonkin_pairs.csv'), index=False)
       
        pd.DataFrame(low_kin_pairs).to_csv(
            os.path.join(output_dir, 'low_similarity_kin_pairs.csv'), index=False)
    except Exception as e:
        raise Exception(f"Error saving results: {str(e)}")

    return kin_similarities, nonkin_similarities

if __name__ == "__main__":
    csv_path = '/cephyr/users/mehdiyev/Alvis/kinship_project/data/processed/fiw/train/filtered_triplets_with_labels.csv'
    output_dir = '/cephyr/users/mehdiyev/Alvis/kinship_project/src/evaluations/cosine_densenet'
    
    # print("\nRunning analysis with sample size of 1000...")
    kin_sims, nonkin_sims = analyze_similarities_batch(
        csv_path=csv_path,
        output_dir=output_dir,
        batch_size=64,
        sample_size=1000  # Set to None to process all triplets
    )