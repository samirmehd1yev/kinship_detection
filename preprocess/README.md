# Data Preprocessing Pipeline

## Overview
This directory contains preprocessing scripts and notebooks for the kinship recognition project using the FIW dataset.

## Files Structure

### Jupyter Notebooks
- `1. analyze_dataset.ipynb`: Dataset statistics and visualization, including face detection/alignment analysis
- `2. process.ipynb`: Primary preprocessing pipeline for face detection/alignment and relationship validation
- `3. individual_check.ipynb`: Quality control script for face embeddings and image filtering
- `4. generate_triplets.ipynb`: Triplet generation from filtered images with relationship validation
- `5. dataset_split.ipynb`: Creates train/validation/test splits with stratification by relationship types

### Python Scripts
- `dataset_clean_hand.py`: Efficient triplet processing script that removes low-quality images
- `enhanced_dataset_splits.py`: Advanced dataset splitting ensuring no family overlap between splits
- `check_family_v2.py`: Validation script for analyzing family distribution across splits
- `fix_relationship_inconsistencies.py`: Script to fix relationship inconsistencies by:
  - Validating relationship types based on gender combinations
  - Correcting multiple relationship assignments for same pairs
  - Using mid.csv files to verify gender information
- `test_unique_members.py`: Analysis script that:
  - Counts unique identities across dataset splits
  - Verifies split integrity
  - Analyzes family member distribution
  - Reports overlap statistics between splits

### Additional Files
- `low_quality_images_hand.txt`: List of manually identified low-quality images
- `logs/`: Contains execution logs with timestamps

## Dependencies
- Python 3.8+
- PyTorch
- InsightFace
- OpenCV
- Pandas
- NumPy
- scikit-learn
- tqdm