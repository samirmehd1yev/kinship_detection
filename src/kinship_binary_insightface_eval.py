import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx2torch import convert
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import time
from pathlib import Path

class KinshipVerificationModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        onnx_model = onnx.load(onnx_path)
        self.backbone = convert(onnx_model)
        self.backbone.requires_grad_(True)
        
        self.use_gradient_checkpointing = False
    
    def forward_one(self, x):
        emb = self.backbone(x)
        return emb
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

class KinshipDataset(Dataset):
    def __init__(self, triplets_df, transform=None, is_training=True):
        self.triplets_df = triplets_df
        self.is_training = is_training
        
        for col in ['Anchor', 'Positive', 'Negative']:
            self.triplets_df[col] = self.triplets_df[col].str.replace(
                '../data',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
                regex=False
            )
        
        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    transforms.RandomErasing(p=0.1)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.triplets_df) * 2
    
    def load_image(self, image_path):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)
                return img
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.1)
    
    def __getitem__(self, idx):
        row_idx = idx // 2
        is_positive = idx % 2 == 0
        
        row = self.triplets_df.iloc[row_idx]
        
        try:
            anchor_img = self.load_image(row['Anchor'])
            
            if is_positive:
                pair_img = self.load_image(row['Positive'])
                label = 1
            else:
                pair_img = self.load_image(row['Negative'])
                label = 0
            
            return {
                'anchor': anchor_img,
                'pair': pair_img,
                'is_kin': torch.LongTensor([label])
            }
        except Exception as e:
            print(f"Error loading images for row {row_idx}: {str(e)}")
            raise

def analyze_thresholds(similarities, labels, num_thresholds=100):
    """Analyze model performance across different thresholds"""
    thresholds = np.linspace(similarities.min(), similarities.max(), num_thresholds)
    threshold_metrics = []
    
    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        threshold_metrics.append({
            'threshold': float(threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'accuracy': float(accuracy)
        })
    
    return threshold_metrics

def plot_evaluation_curves(results, save_dir, split_name):
    """Generate and save evaluation curves"""
    # ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(results['curves']['roc']['fpr'], 
            results['curves']['roc']['tpr'], 
            label=f'ROC curve (AUC = {results["metrics"]["auc_score"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{split_name}_roc_curve.png'))
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    plt.plot(results['curves']['pr']['recall'], 
            results['curves']['pr']['precision'],
            label=f'PR curve (AP = {results["metrics"]["average_precision"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {split_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{split_name}_pr_curve.png'))
    plt.close()

def plot_confusion_matrix(conf_matrix, save_dir, split_name):
    """Generate and save confusion matrix visualization"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Kin', 'Kin'],
                yticklabels=['Non-Kin', 'Kin'])
    plt.title(f'Confusion Matrix - {split_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'{split_name}_confusion_matrix.png'))
    plt.close()

def plot_rtype_evaluation(rtype, metrics, save_dir, split_name):
    """Generate relationship-type specific evaluation plots"""
    rtype_dir = os.path.join(save_dir, 'relationship_types', rtype)
    os.makedirs(rtype_dir, exist_ok=True)
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(metrics['curves']['fpr'], 
            metrics['curves']['tpr'], 
            label=f'ROC curve (AUC = {metrics["auc_score"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split_name} - {rtype.upper()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(rtype_dir, f'{split_name}_{rtype}_roc_curve.png'))
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array(metrics['confusion_matrix']), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Kin', 'Kin'],
                yticklabels=['Non-Kin', 'Kin'])
    plt.title(f'Confusion Matrix - {split_name} - {rtype.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(rtype_dir, f'{split_name}_{rtype}_confusion_matrix.png'))
    plt.close()

def plot_relationship_comparison(all_results, save_dir):
    """Generate comparison plots across relationship types"""
    metrics_dir = os.path.join(save_dir, 'relationship_comparisons')
    os.makedirs(metrics_dir, exist_ok=True)
    
    for split_name, results in all_results.items():
        if 'relationship_metrics' in results:
            # Prepare data for comparison
            rtypes = []
            aucs = []
            accuracies = []
            sample_counts = []
            
            for rtype, metrics in results['relationship_metrics'].items():
                rtypes.append(rtype.upper())
                aucs.append(metrics['auc_score'])
                accuracies.append(metrics['classification_report']['accuracy'])
                sample_counts.append(metrics['sample_count'])
            
            # Plot AUC comparison
            plt.figure(figsize=(12, 6))
            bars = plt.bar(rtypes, aucs)
            plt.title(f'AUC Score by Relationship Type - {split_name}')
            plt.xlabel('Relationship Type')
            plt.ylabel('AUC Score')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(metrics_dir, f'{split_name}_auc_comparison.png'))
            plt.close()
            
            # Plot Accuracy comparison
            plt.figure(figsize=(12, 6))
            bars = plt.bar(rtypes, accuracies)
            plt.title(f'Accuracy by Relationship Type - {split_name}')
            plt.xlabel('Relationship Type')
            plt.ylabel('Accuracy')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(metrics_dir, f'{split_name}_accuracy_comparison.png'))
            plt.close()
            
            # Plot Sample Count comparison
            plt.figure(figsize=(12, 6))
            bars = plt.bar(rtypes, sample_counts)
            plt.title(f'Sample Count by Relationship Type - {split_name}')
            plt.xlabel('Relationship Type')
            plt.ylabel('Number of Samples')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(metrics_dir, f'{split_name}_sample_count.png'))
            plt.close()

def plot_combined_relationship_metrics(all_results, save_dir):
    """Create combined plots showing relationship type performance across all splits"""
    metrics_dir = os.path.join(save_dir, 'combined_comparisons')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Prepare data for plotting
    rtypes = ['ss', 'bb', 'ms', 'fs', 'fd', 'md', 'sibs']
    splits = list(all_results.keys())
    metrics = ['auc_score', 'accuracy']
    metric_names = {'auc_score': 'AUC', 'accuracy': 'Accuracy'}
    
    # Plot settings
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 8*n_metrics))
    bar_width = 0.25
    opacity = 0.8
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for i, split in enumerate(splits):
            metric_values = []
            for rtype in rtypes:
                if rtype in all_results[split]['relationship_metrics']:
                    if metric == 'accuracy':
                        value = all_results[split]['relationship_metrics'][rtype]['classification_report']['accuracy']
                    else:
                        value = all_results[split]['relationship_metrics'][rtype][metric]
                    metric_values.append(value)
                else:
                    metric_values.append(0)
            
            positions = np.arange(len(rtypes)) + i * bar_width
            bars = ax.bar(positions, metric_values, bar_width,
                         alpha=opacity,
                         label=split.capitalize())
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', rotation=90)
        
        # Customize plot
        ax.set_ylabel(metric_names[metric])
        ax.set_title(f'{metric_names[metric]} by Relationship Type Across Splits')
        ax.set_xticks(np.arange(len(rtypes)) + bar_width)
        ax.set_xticklabels([rt.upper() for rt in rtypes])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'combined_metrics_comparison.png'), bbox_inches='tight')
    plt.close()

def format_classification_report(report):
    """Format classification report for text output"""
    formatted = ""
    headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    row_format = "{:>12}" * len(headers) + "\n"
    
    formatted += "-" * 60 + "\n"
    
    for label in ["0", "1"]:
        if label in report:
            metrics = report[label]
            row = [
                label,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']}"
            ]
            formatted += row_format.format(*row)
    
    formatted += "-" * 60 + "\n"
    if 'accuracy' in report:
        formatted += f"Accuracy: {report['accuracy']:.3f}\n"
    
    return formatted

def save_detailed_text_report(all_results, save_dir):
    """Save a detailed text report of all metrics"""
    report_path = os.path.join(save_dir, 'detailed_evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("KINSHIP VERIFICATION MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for split, results in all_results.items():
            f.write("-" * 40 + "\n")
            f.write(f"{split.upper()} SPLIT RESULTS\n")
            f.write("-" * 40 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write(f"AUC Score: {results['metrics']['auc_score']:.4f}\n")
            f.write(f"Average Precision: {results['metrics']['average_precision']:.4f}\n")
            f.write(f"Optimal Threshold: {results['metrics']['optimal_threshold']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(format_classification_report(results['metrics']['classification_report']))
            f.write("\n")
            
            # Per-relationship metrics
            f.write("\nPER-RELATIONSHIP TYPE METRICS:\n")
            f.write("-" * 30 + "\n")
            
            for rtype, metrics in results['relationship_metrics'].items():
                f.write(f"\n{rtype.upper()}:\n")
                f.write(f"Sample Count: {metrics['sample_count']}\n")
                f.write(f"AUC Score: {metrics['auc_score']:.4f}\n")
                f.write(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(format_classification_report(metrics['classification_report']))
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")

def detailed_evaluate_model(model, data_loader, device, save_dir, split_name, split_df):
    """Perform detailed evaluation of the model and save comprehensive results"""
    model.eval()
    all_similarities = []
    all_labels = []
    
    # Create directory for saving results
    os.makedirs(save_dir, exist_ok=True)
    
    RELATIONSHIP_TYPES = ['ss', 'bb', 'ms', 'fs', 'fd', 'md', 'sibs']
    type_indices = {rtype: [] for rtype in RELATIONSHIP_TYPES}
    current_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f'Evaluating {split_name}')):
            anchor = batch['anchor'].to(device)
            pair = batch['pair'].to(device)
            is_kin = batch['is_kin'].squeeze()
            
            emb1, emb2 = model(anchor, pair)
            similarity = F.cosine_similarity(
                F.normalize(emb1, p=2, dim=1),
                F.normalize(emb2, p=2, dim=1)
            )
            
            all_similarities.extend(similarity.cpu().numpy())
            all_labels.extend(is_kin.numpy())
            
            # Track indices for relationship types
            batch_size = len(is_kin)
            for i in range(batch_size):
                idx = current_idx + i
                true_idx = idx // 2  # Get back to the original dataframe index
                if idx % 2 == 0:  # Only process once per pair
                    rtype = split_df.iloc[true_idx]['ptype']
                    if rtype in RELATIONSHIP_TYPES:
                        type_indices[rtype].append(idx)
                        type_indices[rtype].append(idx + 1)  # Include both positive and negative samples
            
            current_idx += batch_size
    
    similarities = np.array(all_similarities)
    labels = np.array(all_labels)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    auc_score = roc_auc_score(labels, similarities)
    
    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, similarities)
    ap_score = average_precision_score(labels, similarities)
    
    # Find optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    predictions = (similarities >= optimal_threshold).astype(int)
    conf_matrix = confusion_matrix(labels, predictions)
    classification_rep = classification_report(labels, predictions, output_dict=True)
    
    # Calculate per-relationship type metrics
    relationship_metrics = {}
    for rtype in RELATIONSHIP_TYPES:
        if type_indices[rtype]:
            rtype_similarities = np.array(all_similarities)[type_indices[rtype]]
            rtype_labels = np.array(all_labels)[type_indices[rtype]]
            
            # Calculate metrics for this relationship type
            rtype_auc = roc_auc_score(rtype_labels, rtype_similarities)
            rtype_fpr, rtype_tpr, rtype_thresholds = roc_curve(rtype_labels, rtype_similarities)
            rtype_optimal_idx = np.argmax(rtype_tpr - rtype_fpr)
            rtype_optimal_threshold = rtype_thresholds[rtype_optimal_idx]
            rtype_predictions = (rtype_similarities >= rtype_optimal_threshold).astype(int)
            rtype_conf_matrix = confusion_matrix(rtype_labels, rtype_predictions)
            rtype_classification_rep = classification_report(rtype_labels, rtype_predictions, output_dict=True)
            
            relationship_metrics[rtype] = {
                'auc_score': float(rtype_auc),
                'optimal_threshold': float(rtype_optimal_threshold),
                'classification_report': rtype_classification_rep,
                'confusion_matrix': rtype_conf_matrix.tolist(),
                'sample_count': len(rtype_labels),
                'curves': {
                    'fpr': rtype_fpr.tolist(),
                    'tpr': rtype_tpr.tolist(),
                    'thresholds': rtype_thresholds.tolist()
                }
            }
            
            # Generate relationship-specific plots
            plot_rtype_evaluation(rtype, relationship_metrics[rtype], save_dir, split_name)

    # Store all results
    results = {
        'split_name': split_name,
        'metrics': {
            'auc_score': float(auc_score),
            'average_precision': float(ap_score),
            'optimal_threshold': float(optimal_threshold),
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist()
        },
        'relationship_metrics': relationship_metrics,
        'curves': {
            'roc': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            },
            'pr': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        },
        'threshold_analysis': analyze_thresholds(similarities, labels)
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(save_dir, f'{split_name}_evaluation_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate and save plots
    plot_evaluation_curves(results, save_dir, split_name)
    plot_confusion_matrix(conf_matrix, save_dir, split_name)
    
    return results

def main():
    # Set up paths and configuration
    checkpoint_path = 'checkpoints/kin_binary_v3/best_model.pth'
    eval_dir = 'evaluations/kinship_binary_insightfacev2'
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load model and checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    model = KinshipVerificationModel(onnx_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load datasets
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand'
    splits = ['train', 'val', 'test']
    
    all_results = {}
    
    for split in splits:
        print(f"\nEvaluating {split} split...")
        # Load data
        data = pd.read_csv(os.path.join(base_path, f'{split}_triplets_enhanced.csv'))
        dataset = KinshipDataset(data, is_training=(split == 'train'))
        dataloader = DataLoader(dataset, 
                              batch_size=128,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)
        
        # Evaluate
        results = detailed_evaluate_model(model, dataloader, device, eval_dir, split, data)
        all_results[split] = results
        
        # Print summary metrics
        print(f"\n{split.capitalize()} Split Summary:")
        print(f"Overall Accuracy: {results['metrics']['classification_report']['accuracy']:.4f}")
        print(f"Overall AUC: {results['metrics']['auc_score']:.4f}")
        print("\nPer-Relationship Type Performance:")
        for rtype, metrics in results['relationship_metrics'].items():
            print(f"\n{rtype.upper()}:")
            print(f"  Accuracy: {metrics['classification_report']['accuracy']:.4f}")
            print(f"  AUC: {metrics['auc_score']:.4f}")
            print(f"  Sample Count: {metrics['sample_count']}")
    
    print("\nGenerating reports and plots...")
    # Generate relationship type comparison plots
    plot_relationship_comparison(all_results, eval_dir)
    
    # Generate combined comparison plots
    plot_combined_relationship_metrics(all_results, eval_dir)
    
    # Save detailed text report
    save_detailed_text_report(all_results, eval_dir)
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_results = {
        'model_info': {
            'checkpoint_path': checkpoint_path,
            'evaluation_timestamp': timestamp
        },
        'results': all_results
    }
    
    with open(os.path.join(eval_dir, f'combined_evaluation_{timestamp}.json'), 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    print("\nEvaluation completed! Results saved in 'evaluations' directory.")
    print("\nCheck the following files for results:")
    print("1. detailed_evaluation_report.txt - Comprehensive text report")
    print("2. combined_comparisons/combined_metrics_comparison.png - Visual comparison across splits")
    print("3. relationship_types/ - Individual relationship type analyses")
    print("4. relationship_comparisons/ - Comparative analysis plots")

if __name__ == "__main__":
    main()

# (kinship_venv_insightface) [mehdiyev@alvis1 src]$ sbatch ../alvis_scripts/kisnhip_training_insighface.sh 
# Submitted batch job 3252651
# (kinship_venv_insightface) [mehdiyev@alvis1 src]$ python kinship_binary_insightfacev2_eval.py 
# /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/src/kinship_binary_insightfacev2_eval.py:532: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   checkpoint = torch.load(checkpoint_path, map_location=device)

# Evaluating train split...
# Evaluating train: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1932/1932 [21:06<00:00,  1.53it/s]

# Train Split Summary:
# Overall Accuracy: 0.8579
# Overall AUC: 0.9326

# Per-Relationship Type Performance:

# SS:
#   Accuracy: 0.8985
#   AUC: 0.9595
#   Sample Count: 49786

# BB:
#   Accuracy: 0.8997
#   AUC: 0.9592
#   Sample Count: 49730

# MS:
#   Accuracy: 0.8684
#   AUC: 0.9459
#   Sample Count: 44550

# FS:
#   Accuracy: 0.8511
#   AUC: 0.9250
#   Sample Count: 32938

# FD:
#   Accuracy: 0.7866
#   AUC: 0.8710
#   Sample Count: 25980

# MD:
#   Accuracy: 0.8143
#   AUC: 0.8934
#   Sample Count: 26840

# SIBS:
#   Accuracy: 0.8464
#   AUC: 0.9209
#   Sample Count: 17368

# Evaluating val split...
# Evaluating val:  93%|███████████████████████████████████████████████████████████████████████████████████████▍      | 385/414 [04:10<00:18,  1.53it/s]Evaluating val: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 414/414 [04:28<00:00,  1.54it/s]

# Val Split Summary:
# Overall Accuracy: 0.7972
# Overall AUC: 0.8719

# Per-Relationship Type Performance:

# SS:
#   Accuracy: 0.8768
#   AUC: 0.9449
#   Sample Count: 9616

# BB:
#   Accuracy: 0.8800
#   AUC: 0.9462
#   Sample Count: 9662

# MS:
#   Accuracy: 0.7196
#   AUC: 0.7855
#   Sample Count: 7018

# FS:
#   Accuracy: 0.7553
#   AUC: 0.8296
#   Sample Count: 6988

# FD:
#   Accuracy: 0.7218
#   AUC: 0.7894
#   Sample Count: 8094

# MD:
#   Accuracy: 0.7880
#   AUC: 0.8648
#   Sample Count: 7580

# SIBS:
#   Accuracy: 0.8654
#   AUC: 0.9357
#   Sample Count: 3914

# Evaluating test split...
# Evaluating test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 408/408 [04:25<00:00,  1.54it/s]

# Test Split Summary:
# Overall Accuracy: 0.7660
# Overall AUC: 0.8426

# Per-Relationship Type Performance:

# SS:
#   Accuracy: 0.8644
#   AUC: 0.9293
#   Sample Count: 6632

# BB:
#   Accuracy: 0.8646
#   AUC: 0.9305
#   Sample Count: 6590

# MS:
#   Accuracy: 0.7086
#   AUC: 0.7850
#   Sample Count: 7804

# FS:
#   Accuracy: 0.7490
#   AUC: 0.8232
#   Sample Count: 9234

# FD:
#   Accuracy: 0.7268
#   AUC: 0.7884
#   Sample Count: 10562

# MD:
#   Accuracy: 0.7680
#   AUC: 0.8454
#   Sample Count: 9462

# SIBS:
#   Accuracy: 0.8116
#   AUC: 0.8779
#   Sample Count: 1868

# Generating reports and plots...

# Evaluation completed! Results saved in 'evaluations' directory.

# Check the following files for results:
# 1. detailed_evaluation_report.txt - Comprehensive text report
# 2. combined_comparisons/combined_metrics_comparison.png - Visual comparison across splits
# 3. relationship_types/ - Individual relationship type analyses
# 4. relationship_comparisons/ - Comparative analysis plots
# (kinship_venv_insightface) [mehdiyev@alvis1 src]$ 