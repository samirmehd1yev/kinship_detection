import pandas as pd
import numpy as np
import os
from collections import defaultdict
from typing import Tuple, Set, Dict, List

def get_family_member_info(path: str) -> Tuple[str, str]:
    """
    Extract family and member information from path
    Args:
        path: Path to image file
    Returns:
        Tuple of (family_id, member_id)
    """
    parts = path.split('/')
    family_id = parts[-3]  # F0001
    member_id = parts[-2]  # MID1
    return family_id, member_id

def verify_relationship_consistency(df: pd.DataFrame) -> Dict[str, Dict[Tuple[str, str], str]]:
    """
    Verify that MID relationships are consistent within families
    Args:
        df: DataFrame containing triplets
    Returns:
        Dictionary of verified family relationships
    """
    family_relationships = defaultdict(dict)
    inconsistencies = []
    
    for _, row in df.iterrows():
        fam1, mid1 = get_family_member_info(row['Anchor'])
        fam2, mid2 = get_family_member_info(row['Positive'])
        rel_type = row['ptype']
        
        if fam1 != fam2:
            print(f"Warning: Cross-family relationship found: {fam1} - {fam2}")
            continue
            
        key = (mid1, mid2)
        rev_key = (mid2, mid1)
        
        if key in family_relationships[fam1]:
            if family_relationships[fam1][key] != rel_type:
                inconsistencies.append((fam1, mid1, mid2, family_relationships[fam1][key], rel_type))
        else:
            family_relationships[fam1][key] = rel_type
    
    if inconsistencies:
        print("\nInconsistent relationships found:")
        for fam, mid1, mid2, rel1, rel2 in inconsistencies:
            print(f"Family {fam}: {mid1}-{mid2} has both {rel1} and {rel2}")
    
    return family_relationships

def analyze_family_structures(df: pd.DataFrame, family_relationships: Dict[str, Dict[Tuple[str, str], str]]) -> Dict[str, Dict[str, str]]:
    """
    Analyze and validate family member roles
    Args:
        df: DataFrame containing triplets
        family_relationships: Dictionary of verified relationships
    Returns:
        Dictionary mapping family members to their roles
    """
    family_roles = defaultdict(dict)
    
    for family, relationships in family_relationships.items():
        member_relationships = defaultdict(list)
        
        # Collect all relationships for each member
        for (mid1, mid2), rel_type in relationships.items():
            member_relationships[mid1].append((mid2, rel_type))
            member_relationships[mid2].append((mid1, rel_type[::-1]))  # Reverse relationship
        
        # Analyze roles based on relationships
        for member, rels in member_relationships.items():
            role_counts = defaultdict(int)
            for _, rel_type in rels:
                if rel_type.startswith('f'):  # father
                    role_counts['father'] += 1
                elif rel_type.startswith('m'):  # mother
                    role_counts['mother'] += 1
                elif rel_type.endswith('s'):  # son
                    role_counts['son'] += 1
                elif rel_type.endswith('d'):  # daughter
                    role_counts['daughter'] += 1
            
            # Assign most frequent role
            if role_counts:
                family_roles[family][member] = max(role_counts.items(), key=lambda x: x[1])[0]
    
    return family_roles

def create_enhanced_splits(triplets_path: str, output_dir: str, train_size: float = 0.7, val_size: float = 0.15) -> Tuple[int, int, int]:
    """
    Create enhanced splits ensuring no family overlap between splits
    Args:
        triplets_path: Path to triplets CSV file
        output_dir: Directory to save split files
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
    Returns:
        Tuple of (train_size, val_size, test_size)
    """
    print("Loading and processing triplets...")
    df = pd.read_csv(triplets_path)
    
    # Remove specific relationship types if needed
    df = df[~df['ptype'].str.contains('gfgs|gfgd|gmgs|gmgd')]
    
    print("Verifying relationship consistency...")
    family_relationships = verify_relationship_consistency(df)
    
    print("Analyzing family structures...")
    family_roles = analyze_family_structures(df, family_relationships)
    
    # Group triplets by family
    family_groups = defaultdict(list)
    for idx, row in df.iterrows():
        family_id, _ = get_family_member_info(row['Anchor'])
        family_groups[family_id].append(idx)
    
    # Sort families by size for better distribution
    family_sizes = [(fam, len(indices)) for fam, indices in family_groups.items()]
    family_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate target sizes
    total_triplets = len(df)
    target_train = total_triplets * train_size
    target_val = total_triplets * val_size
    
    # Distribute families to splits
    train_families = set()
    val_families = set()
    test_families = set()
    
    current_train = 0
    current_val = 0
    
    for family, size in family_sizes:
        if current_train < target_train:
            train_families.add(family)
            current_train += size
        elif current_val < target_val:
            val_families.add(family)
            current_val += size
        else:
            test_families.add(family)
    
    # Create split indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    for family, indices in family_groups.items():
        if family in train_families:
            train_indices.extend(indices)
        elif family in val_families:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)
    
    # Create final dataframes
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    # Print statistics
    print("\nSplit Statistics:")
    print(f"Total triplets: {len(df)}")
    print(f"Train triplets: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Val triplets: {len(val_df)} ({len(val_df)/len(df)*100:.2f}%)")
    print(f"Test triplets: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")
    
    print("\nFamily Statistics:")
    print(f"Train families: {len(train_families)}")
    print(f"Val families: {len(val_families)}")
    print(f"Test families: {len(test_families)}")
    
    # Verify relationship distribution
    def get_rel_dist(df):
        return df['ptype'].value_counts(normalize=True)
    
    print("\nRelationship Distribution:")
    total_dist = get_rel_dist(df)
    train_dist = get_rel_dist(train_df)
    val_dist = get_rel_dist(val_df)
    test_dist = get_rel_dist(test_df)
    
    print("\nRelType\tTotal\tTrain\tVal\tTest")
    for rel in sorted(total_dist.index):
        print(f"{rel}\t{total_dist[rel]:.3f}\t{train_dist.get(rel, 0):.3f}\t"
              f"{val_dist.get(rel, 0):.3f}\t{test_dist.get(rel, 0):.3f}")
    
    # Verify family integrity
    train_fams = {get_family_member_info(row['Anchor'])[0] for _, row in train_df.iterrows()}
    val_fams = {get_family_member_info(row['Anchor'])[0] for _, row in val_df.iterrows()}
    test_fams = {get_family_member_info(row['Anchor'])[0] for _, row in test_df.iterrows()}
    
    print("\nVerifying family split integrity:")
    print(f"Train-Val overlap: {len(train_fams.intersection(val_fams))} (should be 0)")
    print(f"Train-Test overlap: {len(train_fams.intersection(test_fams))} (should be 0)")
    print(f"Val-Test overlap: {len(val_fams.intersection(test_fams))} (should be 0)")
    
    # Save splits
    print("\nSaving splits...")
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train_triplets_enhanced.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_triplets_enhanced.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_triplets_enhanced.csv'), index=False)
    
    return len(train_df), len(val_df), len(test_df)

if __name__ == "__main__":
    triplets_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels_gender_corrected.csv'
    output_dir = '../data/processed/fiw/train/splits_no_overlap_hand'
    
    # Create enhanced splits
    train_size, val_size, test_size = create_enhanced_splits(triplets_path, output_dir)
    
    print(f"\nFinal split sizes:")
    print(f"Train: {train_size}")
    print(f"Validation: {val_size}")
    print(f"Test: {test_size}")

# (kinship_venv_insightface) [mehdiyev@alvis1 notebooks]$ python enhanced_dataset_splits.py 
# Loading and processing triplets...
# Verifying relationship consistency...
# Analyzing family structures...

# Split Statistics:
# Total triplets: 175680
# Train triplets: 123041 (70.04%)
# Val triplets: 26410 (15.03%)
# Test triplets: 26229 (14.93%)

# Family Statistics:
# Train families: 118
# Val families: 106
# Test families: 316

# Relationship Distribution:

# RelType Total   Train   Val     Test
# bb      0.143   0.162   0.124   0.076
# fd      0.127   0.106   0.152   0.201
# fs      0.140   0.134   0.132   0.177
# md      0.124   0.109   0.138   0.184
# ms      0.168   0.180   0.135   0.148
# sibs    0.197   0.211   0.221   0.106
# ss      0.101   0.100   0.098   0.108

# Verifying family split integrity:
# Train-Val overlap: 0 (should be 0)
# Train-Test overlap: 0 (should be 0)
# Val-Test overlap: 0 (should be 0)

# Saving splits...

# Final split sizes:
# Train: 123041
# Validation: 26410
# Test: 26229
# (kinship_venv_insightface) [mehdiyev@alvis1 notebooks]$ 