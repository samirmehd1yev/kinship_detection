import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_enhanced_splits(triplets_path, output_dir, train_size=0.7, val_size=0.15):
    """
    Create enhanced splits with balanced family and relationship distributions
    """
    # Read the filtered triplets
    df = pd.read_csv(triplets_path)
    
    # remove gfgs, gfgd, gmgs, gmgd ptype
    df = df[~df['ptype'].str.contains('gfgs|gfgd|gmgs|gmgd')]
    
    # Extract family information
    def get_family(path):
        return os.path.basename(os.path.dirname(os.path.dirname(path)))
    
    df['anchor_family'] = df['Anchor'].apply(get_family)
    
    # Get relationship types
    df['relationship'] = df['ptype'].str.strip('*')
    
    # Group by family to maintain family-wise splits
    families = df['anchor_family'].unique()
    
    # Stratified split maintaining family and relationship distributions
    train_families, temp_families = train_test_split(
        families,
        train_size=train_size,
        random_state=42
    )
    
    val_size_adjusted = val_size / (1 - train_size)
    val_families, test_families = train_test_split(
        temp_families,
        train_size=val_size_adjusted,
        random_state=42
    )
    
    # Create splits
    train_df = df[df['anchor_family'].isin(train_families)]
    val_df = df[df['anchor_family'].isin(val_families)]
    test_df = df[df['anchor_family'].isin(test_families)]
    
    # Print statistics
    print("\nSplit Statistics:")
    print(f"Total triplets: {len(df)}")
    print(f"Train triplets: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Val triplets: {len(val_df)} ({len(val_df)/len(df)*100:.2f}%)")
    print(f"Test triplets: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")
    
    print("\nFamily Distribution:")
    print(f"Total families: {len(families)}")
    print(f"Train families: {len(train_families)}")
    print(f"Val families: {len(val_families)}")
    print(f"Test families: {len(test_families)}")
    
    # Relationship type distribution
    print("\nRelationship Distribution in Train:")
    print(train_df['relationship'].value_counts(normalize=True))
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train_triplets_enhanced.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_triplets_enhanced.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_triplets_enhanced.csv'), index=False)
    
    # Analyze identity overlap
    def get_identities(df):
        identities = set()
        for path in df['Anchor']:
            family = get_family(path)
            mid = os.path.basename(os.path.dirname(path))
            identities.add(f"{family}/{mid}")
        for path in df['Positive']:
            family = get_family(path)
            mid = os.path.basename(os.path.dirname(path))
            identities.add(f"{family}/{mid}")
        return identities
    
    train_ids = get_identities(train_df)
    val_ids = get_identities(val_df)
    test_ids = get_identities(test_df)
    
    print("\nIdentity Statistics:")
    print(f"Train identities: {len(train_ids)}")
    print(f"Val identities: {len(val_ids)}")
    print(f"Test identities: {len(test_ids)}")
    print(f"Train-Val overlap: {len(train_ids.intersection(val_ids))}")
    print(f"Train-Test overlap: {len(train_ids.intersection(test_ids))}")
    
    # Return the number of unique identities for model config
    all_identities = train_ids.union(val_ids).union(test_ids)
    return len(all_identities)

if __name__ == "__main__":
    triplets_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/filtered_triplets_with_labels.csv'
    output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create enhanced splits
    num_classes = create_enhanced_splits(triplets_path, output_dir)
    print(f"\nNumber of classes for training: {num_classes}")
    
    # Save configuration template
    config_template = f"""
    # Configuration for Enhanced Kinship Model
    input_size = 112
    face_embedding_size = 512
    num_classes = {num_classes}
    
    # Paths for training code
    train_path = '{os.path.join(output_dir, "train_triplets_enhanced.csv")}'
    val_path = '{os.path.join(output_dir, "val_triplets_enhanced.csv")}'
    test_path = '{os.path.join(output_dir, "test_triplets_enhanced.csv")}'
    """
    
    with open(os.path.join(output_dir, 'config_template.txt'), 'w') as f:
        f.write(config_template)