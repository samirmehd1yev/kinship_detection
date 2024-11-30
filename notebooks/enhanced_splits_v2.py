import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict, deque

def find_connected_components(identity_pairs):
    """Find all connected components (families) using BFS."""
    components = []
    visited = set()
    
    # Create adjacency list
    graph = defaultdict(set)
    for id1, id2 in identity_pairs:
        graph[id1].add(id2)
        graph[id2].add(id1)
    
    def bfs(start):
        component = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in component:
                component.add(node)
                queue.extend(graph[node] - component)
        return component
    
    # Find all components
    for identity in graph:
        if identity not in visited:
            component = bfs(identity)
            components.append(component)
            visited.update(component)
    
    return components

def create_enhanced_splits(triplets_path, output_dir, train_size=0.7, val_size=0.15):
    """
    Create enhanced splits ensuring no identity overlap between splits
    """
    print("Loading and processing triplets...")
    df = pd.read_csv(triplets_path)
    
    # Remove specific relationship types if needed
    df = df[~df['ptype'].str.contains('gfgs|gfgd|gmgs|gmgd')]
    
    print("Creating identity mappings...")
    
    def get_identity_key(path):
        family = os.path.basename(os.path.dirname(os.path.dirname(path)))
        person_id = os.path.basename(os.path.dirname(path))
        return f"{family}/{person_id}"
    
    # Create identity pairs and mapping to triplets
    identity_pairs = set()
    identity_to_triplets = defaultdict(set)
    
    for idx, row in df.iterrows():
        anchor_id = get_identity_key(row['Anchor'])
        positive_id = get_identity_key(row['Positive'])
        
        identity_pairs.add((anchor_id, positive_id))
        identity_to_triplets[anchor_id].add(idx)
        identity_to_triplets[positive_id].add(idx)
    
    print("Finding connected components...")
    components = find_connected_components(identity_pairs)
    print(f"Found {len(components)} connected components")
    
    # Sort components by size (number of triplets) for better distribution
    component_sizes = []
    for component in components:
        triplets = set()
        for identity in component:
            triplets.update(identity_to_triplets[identity])
        component_sizes.append((component, len(triplets)))
    
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate target sizes
    total_triplets = len(df)
    target_train = total_triplets * train_size
    target_val = total_triplets * val_size
    
    # Distribute components to splits
    train_components = set()
    val_components = set()
    test_components = set()
    
    current_train = 0
    current_val = 0
    current_test = 0
    
    # Assign components to splits while trying to maintain target ratios
    for component, size in component_sizes:
        train_ratio = current_train / total_triplets if total_triplets > 0 else float('inf')
        val_ratio = current_val / total_triplets if total_triplets > 0 else float('inf')
        test_ratio = current_test / total_triplets if total_triplets > 0 else float('inf')
        
        if train_ratio < train_size:
            train_components.update(component)
            current_train += size
        elif val_ratio < val_size:
            val_components.update(component)
            current_val += size
        else:
            test_components.update(component)
            current_test += size
    
    print("Assigning triplets to splits...")
    train_triplets = set()
    val_triplets = set()
    test_triplets = set()
    
    # Assign triplets based on component membership
    for idx, row in df.iterrows():
        anchor_id = get_identity_key(row['Anchor'])
        positive_id = get_identity_key(row['Positive'])
        
        if anchor_id in train_components or positive_id in train_components:
            train_triplets.add(idx)
        elif anchor_id in val_components or positive_id in val_components:
            val_triplets.add(idx)
        else:
            test_triplets.add(idx)
    
    # Create final dataframes
    train_df = df.loc[list(train_triplets)].copy()
    val_df = df.loc[list(val_triplets)].copy()
    test_df = df.loc[list(test_triplets)].copy()
    
    # Print statistics
    print("\nSplit Statistics:")
    print(f"Total triplets: {len(df)}")
    print(f"Train triplets: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Val triplets: {len(val_df)} ({len(val_df)/len(df)*100:.2f}%)")
    print(f"Test triplets: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")
    
    print("\nIdentity Statistics:")
    train_ids = {get_identity_key(row['Anchor']) for _, row in train_df.iterrows()} | {get_identity_key(row['Positive']) for _, row in train_df.iterrows()}
    val_ids = {get_identity_key(row['Anchor']) for _, row in val_df.iterrows()} | {get_identity_key(row['Positive']) for _, row in val_df.iterrows()}
    test_ids = {get_identity_key(row['Anchor']) for _, row in test_df.iterrows()} | {get_identity_key(row['Positive']) for _, row in test_df.iterrows()}
    
    print(f"Train identities: {len(train_ids)}")
    print(f"Val identities: {len(val_ids)}")
    print(f"Test identities: {len(test_ids)}")
    
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
    
    print("\nVerifying splits integrity:")
    print(f"Train-Val overlap: {len(train_ids.intersection(val_ids))} (should be 0)")
    print(f"Train-Test overlap: {len(train_ids.intersection(test_ids))} (should be 0)")
    print(f"Val-Test overlap: {len(val_ids.intersection(test_ids))} (should be 0)")
    
    # Save splits
    print("\nSaving splits...")
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train_triplets_enhanced.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_triplets_enhanced.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_triplets_enhanced.csv'), index=False)
    
    return len(train_df), len(val_df), len(test_df)

if __name__ == "__main__":
    triplets_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv'
    output_dir = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand'
    
    # Create enhanced splits
    train_size, val_size, test_size = create_enhanced_splits(triplets_path, output_dir)
    
    print(f"\nFinal split sizes:")
    print(f"Train: {train_size}")
    print(f"Validation: {val_size}")
    print(f"Test: {test_size}")


#     (kinship_venv) [mehdiyev@alvis1 src]$ cd ../notebooks/
# (kinship_venv) [mehdiyev@alvis1 notebooks]$ python enhanced_splits_v2.py 
# Loading and processing triplets...
# Creating identity mappings...
# Finding connected components...
# Found 547 connected components
# Assigning triplets to splits...

# Split Statistics:
# Total triplets: 176108
# Train triplets: 123596 (70.18%)
# Val triplets: 26436 (15.01%)
# Test triplets: 26076 (14.81%)

# Identity Statistics:
# Train identities: 679
# Val identities: 466
# Test identities: 1084

# Relationship Distribution:

# RelType Total   Train   Val     Test
# bb      0.187   0.201   0.183   0.126
# fd      0.127   0.105   0.153   0.203
# fs      0.140   0.133   0.132   0.177
# md      0.125   0.109   0.143   0.181
# ms      0.169   0.180   0.133   0.150
# sibs    0.066   0.070   0.074   0.036
# ss      0.187   0.201   0.182   0.127

# Verifying splits integrity:
# Train-Val overlap: 0 (should be 0)
# Train-Test overlap: 0 (should be 0)
# Val-Test overlap: 0 (should be 0)

# Saving splits...

# Final split sizes:
# Train: 123596
# Validation: 26436
# Test: 26076