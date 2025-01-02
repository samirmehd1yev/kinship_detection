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

### Evaluations
- `evaluations/`: Evaluation results and visualizations:
  - ROC curves
  - Confusion matrices
  - Similarity distributions
  - Performance metrics

### Archived Code
- `archive/`: Previous versions of training scripts and models

### Configuration
- `configs/train_config.json`: Training configuration parameters

### Outputs
- `training_output/`: Training logs and error reports
- `embeddings_output/`: Embedding generation logs and error reports
- `results/`: Model evaluation results

## Pretrained Models
- `pretrained_models/`: Contains pretrained model weights
  - CosFace backbone (Resnet50)

