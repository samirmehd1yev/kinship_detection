import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_curve, 
    roc_curve, 
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from tqdm import tqdm
import json
from datetime import datetime
from kin_nonkin_training import KinshipConfig, KinshipDataset, KinshipModel, create_dataloaders

# Setup evaluation directory
def setup_eval_dirs():
    # Create base evaluation directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('evaluations', f'eval_{timestamp}')
    
    # Create subdirectories
    dirs = {
        'base': base_dir,
        'plots': os.path.join(base_dir, 'plots'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'features': os.path.join(base_dir, 'features')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def evaluate_model_detailed(model, test_loader, device):
    model.eval()
    predictions = []
    labels = []
    probabilities = []
    anchor_features_list = []
    other_features_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)
            
            # Get predictions
            outputs = model(anchor, other)
            kinship_score = outputs['kinship_score']
            
            # Store features
            anchor_features_list.append(outputs['anchor_features'].cpu().numpy())
            other_features_list.append(outputs['other_features'].cpu().numpy())
            
            # Convert logits to probabilities
            kinship_prob = torch.sigmoid(kinship_score)
            
            # Store results
            probabilities.extend(kinship_prob.cpu().numpy())
            predictions.extend((kinship_prob > 0.5).cpu().numpy())
            labels.extend(is_related.cpu().numpy())
    
    return (np.array(predictions), np.array(probabilities), np.array(labels), 
            np.vstack(anchor_features_list), np.vstack(other_features_list))

def plot_confusion_matrix(y_true, y_pred, save_path):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Kin', 'Kin'],
                yticklabels=['Non-Kin', 'Kin'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_prob, save_path):
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pr_auc

def analyze_threshold_impact(y_true, y_prob, save_dir):
    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(12, 8))
    for column in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(metrics_df['threshold'], metrics_df[column], label=column.capitalize(), marker='o')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Classification Threshold on Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'threshold_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(save_dir, 'threshold_metrics.csv'), index=False)
    
    return metrics_df

def analyze_feature_distributions(anchor_features, other_features, labels, save_dir):
    # Calculate feature similarities
    similarities = np.sum(anchor_features * other_features, axis=1)
    
    # Plot distribution of similarities
    plt.figure(figsize=(12, 8))
    plt.hist(similarities[labels == 1], alpha=0.5, label='Kin', bins=50, color='green')
    plt.hist(similarities[labels == 0], alpha=0.5, label='Non-kin', bins=50, color='red')
    plt.xlabel('Feature Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Feature Similarities for Kin vs Non-kin Pairs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save feature analysis
    feature_stats = {
        'kin_mean_similarity': float(np.mean(similarities[labels == 1])),
        'nonkin_mean_similarity': float(np.mean(similarities[labels == 0])),
        'kin_std_similarity': float(np.std(similarities[labels == 1])),
        'nonkin_std_similarity': float(np.std(similarities[labels == 0]))
    }
    
    with open(os.path.join(save_dir, 'feature_stats.json'), 'w') as f:
        json.dump(feature_stats, f, indent=4)
    
    return similarities, feature_stats

def run_full_evaluation(model, test_loader, device):
    # Setup directories
    eval_dirs = setup_eval_dirs()
    
    print("Starting model evaluation...")
    
    # Get predictions and features
    predictions, probabilities, labels, anchor_features, other_features = evaluate_model_detailed(model, test_loader, device)
    
    # Save classification report
    report = classification_report(labels, predictions, output_dict=True)
    with open(os.path.join(eval_dirs['metrics'], 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions))
    
    # Generate and save plots
    plot_confusion_matrix(labels, predictions, 
                         os.path.join(eval_dirs['plots'], 'confusion_matrix.png'))
    
    roc_auc = plot_roc_curve(labels, probabilities, 
                            os.path.join(eval_dirs['plots'], 'roc_curve.png'))
    
    pr_auc = plot_precision_recall_curve(labels, probabilities, 
                                       os.path.join(eval_dirs['plots'], 'pr_curve.png'))
    
    # Analyze threshold impact
    threshold_metrics = analyze_threshold_impact(labels, probabilities, eval_dirs['metrics'])
    
    # Analyze feature distributions
    similarities, feature_stats = analyze_feature_distributions(
        anchor_features, other_features, labels, eval_dirs['features']
    )
    
    # Save overall metrics
    overall_metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'feature_stats': feature_stats
    }
    
    with open(os.path.join(eval_dirs['metrics'], 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    
    print("\nEvaluation complete! Results saved in:", eval_dirs['base'])
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'labels': labels,
        'threshold_metrics': threshold_metrics,
        'similarities': similarities,
        'overall_metrics': overall_metrics,
        'eval_dirs': eval_dirs
    }

# Load and evaluate model
def evaluate_saved_model(model_path, test_loader):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model from:", model_path)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model (you'll need to import your model class)
    model = KinshipModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run evaluation
    results = run_full_evaluation(model, test_loader, device)
    
    return results

# Example usage
if __name__ == "__main__":
    model_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model/model/best_kin_nonkin_model.pth"
    
    # Create test dataloader (you'll need your dataset setup code)
    config = KinshipConfig()
    test_loader = create_dataloaders(config)[2]
    
    # Run evaluation
    results = evaluate_saved_model(model_path, test_loader)