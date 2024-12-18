import pandas as pd
import os
import logging
import re

def get_gender_from_mid_csv(family, mid):
    """Extract gender from mid.csv"""
    try:
        mid_csv_path = f"/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/raw/fiw/train/train-faces/{family}/mid.csv"
        
        if not os.path.exists(mid_csv_path):
            logging.warning(f"mid.csv not found for {family}")
            return None
            
        mid_df = pd.read_csv(mid_csv_path)
        gender = mid_df[mid_df['MID'] == int(mid)]['Gender'].iloc[0].lower()
        return 'm' if gender in ['m', 'male'] else 'f'
    except Exception as e:
        logging.warning(f"Error getting gender for {family}/MID{mid}: {e}")
        return None

def validate_relationship_type(ptype, p1_gender, p2_gender):
    """Validate if relationship type matches gender combination"""
    relationship_rules = {
        'fs': {('m', 'm')},                     # father-son: must be male->male
        'ms': {('f', 'm'), ('m', 'f')},         # mother-son: can be female->male or male->female
        'ss': {('f', 'f')},                     # sister-sister: must be female->female
        'bb': {('m', 'm')},                     # brother-brother: must be male->male
        'fd': {('m', 'f'), ('f', 'm')},         # father-daughter: can be male->female or female->male
        'md': {('f', 'f')},                     # mother-daughter: must be female->female
        'sibs': {('m', 'f'), ('f', 'm')},       # siblings: can be male->female or female->male
        'gfgs': {('m', 'm')},                   # grandfather-grandson: must be male->male
        'gfgd': {('m', 'f'), ('f', 'm')},       # grandfather-granddaughter: can be male->female or female->male
        'gmgs': {('f', 'm'), ('m', 'f')},       # grandmother-grandson: can be female->male or male->female
        'gmgd': {('f', 'f')}                    # grandmother-granddaughter: must be female->female
    }
    
    gender_pair = (p1_gender, p2_gender)
    return ptype in relationship_rules and gender_pair in relationship_rules[ptype]

def extract_family_mid(path):
    """Extract family and MID from file path"""
    match = re.search(r'F(\d+)/MID(\d+)', path)
    if match:
        return f"F{match.group(1)}", int(match.group(2))
    return None, None

def clean_pairs_and_triplets(pairs_path, triplets_path):
    # Read pairs and triplets CSV
    pairs_df = pd.read_csv(pairs_path)
    triplets_df = pd.read_csv(triplets_path)
    
    # Find pairs with multiple relationship types
    pair_counts = pairs_df.groupby(['p1', 'p2'])['ptype'].nunique()
    multiple_rel_pairs = pair_counts[pair_counts > 1].index
    
    print(f"Found {len(multiple_rel_pairs)} pairs with multiple relationship types")
    
    # Create a mapping for corrections
    corrections_map = {}
    invalid_pairs = []
    
    # Process pairs with multiple relationships
    for p1, p2 in multiple_rel_pairs:
        fam1, mid1 = p1.split('/')
        fam2, mid2 = p2.split('/')
        mid1 = int(mid1[3:])
        mid2 = int(mid2[3:])
        
        p1_gender = get_gender_from_mid_csv(fam1, mid1)
        p2_gender = get_gender_from_mid_csv(fam2, mid2)
        
        if p1_gender and p2_gender:
            rel_types = pairs_df[(pairs_df['p1'] == p1) & (pairs_df['p2'] == p2)]['ptype'].unique()
            
            valid_rel = None
            for rel in rel_types:
                if validate_relationship_type(rel, p1_gender, p2_gender):
                    valid_rel = rel
                    break
            
            key = f"{p1},{p2}"
            if valid_rel:
                corrections_map[key] = valid_rel
            else:
                invalid_pairs.append({
                    'p1': p1,
                    'p2': p2,
                    'p1_gender': p1_gender,
                    'p2_gender': p2_gender,
                    'original_types': list(rel_types)
                })
    
    # Now correct triplets
    corrected_triplets = []
    invalid_triplets = []
    
    for _, triplet in triplets_df.iterrows():
        anchor_fam, anchor_mid = extract_family_mid(triplet['Anchor'])
        pos_fam, pos_mid = extract_family_mid(triplet['Positive'])
        
        if anchor_fam and pos_fam:
            key1 = f"{anchor_fam}/MID{anchor_mid},{pos_fam}/MID{pos_mid}"
            key2 = f"{pos_fam}/MID{pos_mid},{anchor_fam}/MID{anchor_mid}"
            
            # Check if this pair needs correction
            correction = corrections_map.get(key1) or corrections_map.get(key2)
            
            if correction and correction != triplet['ptype']:
                # Create corrected triplet
                corrected_triplet = triplet.copy()
                corrected_triplet['ptype'] = correction
                corrected_triplets.append(corrected_triplet)
                
                # print(f"Correcting Triplet_ID {triplet['Triplet_ID']}:")
                # print(f"Old ptype: {triplet['ptype']}")
                # print(f"New ptype: {correction}")
                # print("---")
            else:
                corrected_triplets.append(triplet)
    
    # Create corrected triplets DataFrame
    corrected_triplets_df = pd.DataFrame(corrected_triplets)
    
    # Save results
    output_path = os.path.splitext(triplets_path)[0] + '_gender_corrected.csv'
    corrected_triplets_df.to_csv(output_path, index=False)
    
    pd.DataFrame(invalid_pairs).to_csv('invalid_pairs.csv', index=False)
    
    print("\nSummary:")
    print(f"Total pairs with multiple relationships: {len(multiple_rel_pairs)}")
    print(f"Invalid pairs found: {len(invalid_pairs)}")
    print(f"Triplets processed: {len(triplets_df)}")
    print(f"Triplets corrected: {len(corrected_triplets_df) - len(triplets_df)}")
    print(f"\nCorrected triplets saved to: {output_path}")

if __name__ == "__main__":
    pairs_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/raw/fiw/train/train-pairs.csv"
    triplets_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv"
    
    clean_pairs_and_triplets(pairs_path, triplets_path)
    
# Summary:
# Total pairs with multiple relationships: 1458
# Invalid pairs found: 0
# Triplets processed: 180677
# Triplets corrected: 0

# Corrected triplets saved to: /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels_gender_corrected.csv