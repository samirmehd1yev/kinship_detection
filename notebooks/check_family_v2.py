import pandas as pd
import os
from collections import defaultdict

def analyze_family_distribution(triplets_path, splits_dir):
    """Analyze family distribution across splits"""
    
    # Read the splits
    train_df = pd.read_csv(os.path.join(splits_dir, 'train_triplets_enhanced.csv'))
    val_df = pd.read_csv(os.path.join(splits_dir, 'val_triplets_enhanced.csv'))
    test_df = pd.read_csv(os.path.join(splits_dir, 'test_triplets_enhanced.csv'))
    
    # Function to get family from path
    def get_family(path):
        return os.path.basename(os.path.dirname(os.path.dirname(path)))
    
    # Get unique families in each split
    def get_families(df):
        families = set()
        for _, row in df.iterrows():
            families.add(get_family(row['Anchor']))
            families.add(get_family(row['Positive']))
        return families
    
    train_families = get_families(train_df)
    val_families = get_families(val_df)
    test_families = get_families(test_df)
    
    # Print statistics
    print("\nFamily Distribution Analysis:")
    print(f"Train families: {len(train_families)}")
    print(f"Val families: {len(val_families)}")
    print(f"Test families: {len(test_families)}")
    
    # Check overlap
    print("\nFamily Overlap Analysis:")
    train_val_overlap = train_families.intersection(val_families)
    train_test_overlap = train_families.intersection(test_families)
    val_test_overlap = val_families.intersection(test_families)
    
    print(f"Train-Val family overlap: {len(train_val_overlap)}")
    print(f"Train-Test family overlap: {len(train_test_overlap)}")
    print(f"Val-Test family overlap: {len(val_test_overlap)}")
    
    if len(train_val_overlap) > 0:
        print("\nOverlapping families between Train-Val:")
        print(sorted(train_val_overlap)[:5], "..." if len(train_val_overlap) > 5 else "")
    
    # Analyze triplets per family
    train_family_sizes = train_df.groupby(train_df['Anchor'].apply(get_family)).size()
    val_family_sizes = val_df.groupby(val_df['Anchor'].apply(get_family)).size()
    test_family_sizes = test_df.groupby(test_df['Anchor'].apply(get_family)).size()
    
    print("\nTriplets per Family Statistics:")
    print("\nTrain split:")
    print(f"Mean triplets per family: {train_family_sizes.mean():.2f}")
    print(f"Min triplets per family: {train_family_sizes.min()}")
    print(f"Max triplets per family: {train_family_sizes.max()}")
    
    print("\nVal split:")
    print(f"Mean triplets per family: {val_family_sizes.mean():.2f}")
    print(f"Min triplets per family: {val_family_sizes.min()}")
    print(f"Max triplets per family: {val_family_sizes.max()}")
    
    print("\nTest split:")
    print(f"Mean triplets per family: {test_family_sizes.mean():.2f}")
    print(f"Min triplets per family: {test_family_sizes.min()}")
    print(f"Max triplets per family: {test_family_sizes.max()}")

if __name__ == "__main__":
    triplets_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/filtered_triplets_with_labels.csv'
    splits_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand'
    
    analyze_family_distribution(triplets_path, splits_dir)
    
# (kinship_venv_insightface) [mehdiyev@alvis1 notebooks]$ python check_family_v2.py 

# Family Distribution Analysis:
# Train families: 118
# Val families: 106
# Test families: 316

# Family Overlap Analysis:
# Train-Val family overlap: 0
# Train-Test family overlap: 0
# Val-Test family overlap: 0

# Triplets per Family Statistics:

# Train split:
# Mean triplets per family: 1042.72
# Min triplets per family: 349
# Max triplets per family: 22641

# Val split:
# Mean triplets per family: 249.15
# Min triplets per family: 187
# Max triplets per family: 348

# Test split:
# Mean triplets per family: 83.00
# Min triplets per family: 3
# Max triplets per family: 184
# (kinship_venv_insightface) [mehdiyev@alvis1 notebooks]$ 