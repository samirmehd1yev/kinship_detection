import os
import numpy as np
import pandas as pd
import torch
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
from tqdm import tqdm  # Ensure tqdm is imported from the correct module
import json
from datetime import datetime
from kin_nonkin_training_v3 import (
    KinshipConfig,
    KinshipDataset,
    KinshipModel,
    create_dataloaders
)

# Setup evaluation directory
def setup_eval_dirs():
    # Create base evaluation directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('evaluations', f'eval_v3_{timestamp}')

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
        for batch in tqdm(test_loader, desc='Evaluating', leave=True):  # Ensure tqdm is visible in terminal
            # Move data to device
            anchor = batch['anchor'].to(device)
            other = batch['other'].to(device)
            is_related = batch['is_related'].to(device)

            # Get outputs
            outputs = model(anchor, other)
            logits = outputs['output']

            # Store features
            anchor_features_list.append(outputs['anchor_features'].cpu().numpy())
            other_features_list.append(outputs['other_features'].cpu().numpy())

            # Convert logits to probabilities using softmax
            probs = torch.softmax(logits, dim=1)
            kinship_prob = probs[:, 1]  # Probability of class '1' (kin)

            # Store results
            probabilities.extend(kinship_prob.cpu().numpy())
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(is_related.cpu().numpy())

    return (
        np.array(predictions),
        np.array(probabilities),
        np.array(labels),
        np.vstack(anchor_features_list),
        np.vstack(other_features_list)
    )

def plot_confusion_matrix(y_true, y_pred, save_path):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Kin', 'Kin'],
                yticklabels=['Non-Kin', 'Kin'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return roc_auc

def plot_precision_recall_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return pr_auc

def analyze_threshold_impact(y_true, y_prob, save_dir):
    thresholds = np.arange(0.0, 1.01, 0.05)
    metrics = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
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
    plt.figure(figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(metrics_df['threshold'], metrics_df[metric], label=metric.capitalize())
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Impact on Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'threshold_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(save_dir, 'threshold_metrics.csv'), index=False)

    return metrics_df

def analyze_feature_distributions(anchor_features, other_features, labels, save_dir):
    # Calculate cosine similarities
    similarities = np.sum(anchor_features * other_features, axis=1)

    # Plot distribution of similarities
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities[labels == 1], label='Kin', color='green', kde=True, stat="density", bins=50)
    sns.histplot(similarities[labels == 0], label='Non-Kin', color='red', kde=True, stat="density", bins=50)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Feature Similarity Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'feature_similarity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save feature statistics
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
    base_dir = eval_dirs['base']

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
    plot_confusion_matrix(labels, predictions, os.path.join(eval_dirs['plots'], 'confusion_matrix.png'))
    roc_auc = plot_roc_curve(labels, probabilities, os.path.join(eval_dirs['plots'], 'roc_curve.png'))
    pr_auc = plot_precision_recall_curve(labels, probabilities, os.path.join(eval_dirs['plots'], 'precision_recall_curve.png'))

    # Analyze threshold impact
    threshold_metrics = analyze_threshold_impact(labels, probabilities, eval_dirs['metrics'])

    # Analyze feature distributions
    similarities, feature_stats = analyze_feature_distributions(anchor_features, other_features, labels, eval_dirs['features'])

    # Save overall metrics
    overall_metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'feature_stats': feature_stats
    }

    with open(os.path.join(eval_dirs['metrics'], 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=4)

    print("\nEvaluation complete! Results saved in:", base_dir)

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
def evaluate_saved_model(model_path):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model checkpoint
    print("Loading model from:", model_path)
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model with the same configuration
    config = KinshipConfig()
    model = KinshipModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create test dataloader
    test_dataset = KinshipDataset(config.test_path, config, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Run full evaluation
    results = run_full_evaluation(model, test_loader, device)

    return results

# Main execution
if __name__ == "__main__":
    model_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model3/model/best_model.pth"

    # Evaluate the saved model on the test dataset
    evaluate_saved_model(model_path)