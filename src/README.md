# Source Code Directory

## Overview
This directory contains all source code for the kinship recognition models, training scripts, evaluations.

## Directory Structure

### Main Training Scripts
- `kinship_binary_insightface_v*.py`: Binary classification models using InsightFace
- `kin_relationship_v*.py`: Relationship type classification models
- `kinship_ensemble_model_v1.py`: Ensemble model implementation
- `test_models.py`: Model testing utilities

### Embeddings Generation
- `embeddings/`: Contains scripts and outputs for different embedding models:
  - InsightFace embeddings
  - ArcFace embeddings
  - FaceNet512 embeddings
  - DeepFace embeddings

### Cosine Similarity Analysis
- `cosine_sim_check/`: Scripts for different model similarity checks:
  - DeepFace
  - DenseNet
  - InceptionResNet/VGG
  - InsightFace

### Model Checkpoints
- `checkpoints/`: Saved model states and results
  - Binary classification models
  - Relationship classification models
  - Ensemble models
- Due to size limits cannot push into github.
- Link: https://drive.google.com/drive/folders/1E0XG8yjgkJhFI1W0dVBPxk0OfWhjG5r8?usp=share_link

### Evaluations
- `evaluations/`: Evaluation results and visualizations:
  - ROC curves
  - Confusion matrices
  - Similarity distributions
  - Performance metrics

### Archived Code
- `archive/`: Contains previous versions of training scripts and models. These early attempts did not perform better than the latest versions.

### Configuration
- `configs/train_config.json`: Training configuration parameters for v2 training code

### Outputs
- `training_output/`: Training logs and error reports
- `embeddings_output/`: Embedding generation logs and error reports
- `results/`: Model evaluation results

## Pretrained Models
- `pretrained_models/`: Contains pretrained model weights
  - CosFace backbone (Resnet50)