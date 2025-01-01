import pandas as pd
import os

# set gpu 2(this is the code for me to use the second gpu)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def analyze_dataset(metadata_path, train_path, val_path, test_path):
    """
    Analyze the dataset to get statistics and number of classes
    """
    # Read metadata
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df.Is_Kept]  # Filter kept images
    
    # Read triplets files
    train_triplets = pd.read_csv(train_path)
    val_triplets = pd.read_csv(val_path)
    test_triplets = pd.read_csv(test_path)
    
    def get_unique_identities(triplets_df):
        unique_identities = set()
        
        # Process Anchor images
        for path in triplets_df['Anchor']:
            # Extract family and MID from path
            # ../data/processed/fiw/train/train-faces/F0001/MID1/P00002_face0.jpg
            family = os.path.basename(os.path.dirname(os.path.dirname(path)))
            mid = os.path.basename(os.path.dirname(path))
            identity = f"{family}/{mid}"
            unique_identities.add(identity)
        
        # Process Positive images
        for path in triplets_df['Positive']:
            family = os.path.basename(os.path.dirname(os.path.dirname(path)))
            mid = os.path.basename(os.path.dirname(path))
            identity = f"{family}/{mid}"
            unique_identities.add(identity)
        
        return unique_identities
    
    # Get unique identities from each split
    train_identities = get_unique_identities(train_triplets)
    val_identities = get_unique_identities(val_triplets)
    test_identities = get_unique_identities(test_triplets)
    
    # Get all unique identities
    all_identities = train_identities.union(val_identities).union(test_identities)
    
    # Get families info
    all_families = metadata_df.Family.nunique()
    train_families = len(set(os.path.basename(os.path.dirname(os.path.dirname(path))) 
                           for path in train_triplets['Anchor']))
    
    print("\nDataset Statistics:")
    print(f"Total unique identities across all splits: {len(all_identities)}")
    print(f"Train split unique identities: {len(train_identities)}")
    print(f"Validation split unique identities: {len(val_identities)}")
    print(f"Test split unique identities: {len(test_identities)}")
    
    print("\nFamily Statistics:")
    print(f"Total families in metadata: {all_families}")
    print(f"Families in train split: {train_families}")
    
    # Identity overlap analysis
    train_val_overlap = len(train_identities.intersection(val_identities))
    train_test_overlap = len(train_identities.intersection(test_identities))
    val_test_overlap = len(val_identities.intersection(test_identities))
    
    print("\nIdentity Overlap Analysis:")
    print(f"Train-Val overlap: {train_val_overlap} identities")
    print(f"Train-Test overlap: {train_test_overlap} identities")
    print(f"Val-Test overlap: {val_test_overlap} identities")
    
    # Get member counts per family
    family_member_counts = metadata_df.groupby('Family').Member.nunique()
    print("\nFamily Member Distribution:")
    print(family_member_counts.describe())
    
    print("\nTop 10 largest families:")
    print(family_member_counts.nlargest(10))
    
    return len(all_identities)  # This will be your num_classes

if __name__ == "__main__":
    # File paths
    metadata_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/fiw_metadata_filtered.csv'
    train_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/train_triplets_enhanced.csv'
    val_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/val_triplets_enhanced.csv'
    test_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_enhanced/test_triplets_enhanced.csv'
    
    # Get number of classes and dataset statistics
    num_classes = analyze_dataset(metadata_path, train_path, val_path, test_path)
    print(f"\nNumber of classes for training: {num_classes}")