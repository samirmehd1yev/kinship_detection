import pandas as pd
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
from functools import lru_cache
from typing import Set, Dict, List, Optional

class TripletProcessor:
    def __init__(self, csv_path: str, bad_images_path: str):
        self.df = pd.read_csv(csv_path)
        self.bad_paths = self._load_bad_paths(bad_images_path)
        self._family_cache: Dict[str, str] = {}
        self._negative_cache: Dict[str, List[str]] = {}
        
    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize path using a more efficient method"""
        return path.replace('../', '').replace('./', '')
    
    def _load_bad_paths(self, bad_images_path: str) -> Set[str]:
        """Load and normalize bad image paths"""
        with open(bad_images_path, 'r') as f:
            return {self.normalize_path(line.strip()) 
                   for line in f 
                   if line.strip()}
    
    @lru_cache(maxsize=1024)
    def _extract_family(self, path: str) -> str:
        """Extract family ID from path with caching"""
        return path.split('/F')[1][:4]
    
    def _get_negative_candidates(self, current_family: str) -> List[str]:
        """Get and cache negative candidates for a family"""
        if current_family not in self._negative_cache:
            family_pattern = f'F{current_family}/'
            other_families = self.df[
                ~self.df['Anchor'].str.contains(family_pattern) &
                ~self.df['Positive'].str.contains(family_pattern) &
                ~self.df['Negative'].str.contains(family_pattern)
            ]
            self._negative_cache[current_family] = np.concatenate((
                other_families['Anchor'].unique(),
                other_families['Positive'].unique()
            )).tolist()
        return self._negative_cache[current_family]
    
    def get_random_negative(self, current_family: str, exclude_paths: Set[str]) -> Optional[str]:
        """Get random negative image with optimized filtering"""
        candidates = self._get_negative_candidates(current_family)
        valid_negatives = [neg for neg in candidates 
                         if self.normalize_path(neg) not in exclude_paths]
        return random.choice(valid_negatives) if valid_negatives else None
    
    def process(self, output_path: str) -> None:
        """Process triplets with vectorized operations where possible"""
        initial_count = len(self.df)
        print(f"Processing {initial_count} triplets...")
        
        # Vectorized normalization of paths
        anchor_norms = self.df['Anchor'].apply(self.normalize_path)
        positive_norms = self.df['Positive'].apply(self.normalize_path)
        negative_norms = self.df['Negative'].apply(self.normalize_path)
        
        # Create masks for valid anchor/positive pairs and negative paths
        valid_anchor_positive = (~anchor_norms.isin(self.bad_paths)) & (~positive_norms.isin(self.bad_paths))
        needs_new_negative = negative_norms.isin(self.bad_paths)
        
        # Combine conditions: keep rows where anchor/positive are valid
        valid_indices = valid_anchor_positive[valid_anchor_positive].index
        needs_replacement = needs_new_negative[valid_indices]
        
        # Initialize filtered rows with pre-allocated list
        filtered_rows = []
        filtered_rows_append = filtered_rows.append  # Local reference for faster append
        for idx in tqdm(valid_indices):
            row = self.df.iloc[idx]
            if negative_norms[idx] in self.bad_paths:
                current_family = self._extract_family(row['Anchor'])
                new_negative = self.get_random_negative(current_family, self.bad_paths)
                if new_negative:
                    row = row.copy()
                    row['Negative'] = new_negative
            filtered_rows_append(row)
        
        # Create and save filtered DataFrame
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.to_csv(output_path, index=False)
        
        # Print statistics
        final_count = len(filtered_df)
        removed_count = initial_count - final_count
        print("\nFiltering Results:")
        print(f"Initial triplets: {initial_count:,}")
        print(f"Removed triplets: {removed_count:,}")
        print(f"Final triplets: {final_count:,}")
        print(f"Removal percentage: {(removed_count/initial_count)*100:.2f}%")
        print(f"\nFiltered dataset saved to: {output_path}")

def main():
    csv_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/filtered_triplets_with_labels.csv'
    bad_images_path = 'low_quality_images_hand.txt'
    output_path = '../data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv'
    
    processor = TripletProcessor(csv_path, bad_images_path)
    processor.process(output_path)

if __name__ == "__main__":
    main()

# This script is used remove bad images from the dataset. Bad images are chsoen by hand with manual inspection.
# in low_quality_images_hand.txt file, we have paths of bad images. We will remove these images from the dataset.

# Output:
# (kinship_venv_insightface) [mehdiyev@alvis1 notebooks]$ python dataset_clean_hand.py 
# Processing 189550 triplets...
# 100%|█████████████████████████████████████████████████████████████████████████████████████| 180677/180677 [02:54<00:00, 1037.36it/s]

# Filtering Results:
# Initial triplets: 189,550
# Removed triplets: 8,873
# Final triplets: 180,677
# Removal percentage: 4.68%

# Filtered dataset saved to: ../data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv