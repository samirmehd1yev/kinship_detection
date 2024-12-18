import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# sue gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def process_image_file(face_app, image_path):
    """Process a single image file and return age and gender predictions."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
            
        # Detect and analyze faces
        faces = face_app.get(img)
        
        if not faces:
            print(f"No face detected in: {image_path}")
            return None
            
        # Use the largest face if multiple faces are detected
        if len(faces) > 1:
            # Get the face with largest bounding box area
            areas = [(face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) for face in faces]
            face = faces[np.argmax(areas)]
        else:
            face = faces[0]
            
        return {
            'age': float(face.age),
            'gender': int(face.gender)  # 1 for male, 0 for female
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def extract_and_save_metadata(data_path, output_file):
    """
    Extract age and gender features from all images in the dataset and save to JSON.
    """
    # Initialize face analysis
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(128, 128))
    
    # Get all triplets files
    splits = ['train', 'val', 'test']
    all_images = set()
    
    for split in splits:
        df = pd.read_csv(os.path.join(data_path, f'{split}_triplets_enhanced.csv'))
        all_images.update(df['Anchor'].tolist())
        all_images.update(df['Positive'].tolist())
        all_images.update(df['Negative'].tolist())
    
    print(f"Total unique images to process: {len(all_images)}")
    
    # Process all images
    metadata_dict = {}
    failed_images = []
    
    for image_path in tqdm(all_images, desc="Processing images"):
        metadata = process_image_file(face_app, image_path)
        if metadata is not None:
            metadata_dict[image_path] = metadata
        else:
            failed_images.append(image_path)
    
    # Save results
    print(f"\nSuccessfully processed {len(metadata_dict)} images")
    print(f"Failed to process {len(failed_images)} images")
    print(f"Saving results to {output_file}")
    
    # Save main metadata
    with open(output_file, 'w') as f:
        json.dump(metadata_dict, f, indent=4)
    
    # Save list of failed images
    failed_file = output_file.replace('.json', '_failed.txt')
    with open(failed_file, 'w') as f:
        for img_path in failed_images:
            f.write(f"{img_path}\n")
    
    # Print statistics
    ages = [m['age'] for m in metadata_dict.values()]
    genders = [m['gender'] for m in metadata_dict.values()]
    
    print("\nDataset Statistics:")
    print(f"Age - Mean: {np.mean(ages):.2f}, Std: {np.std(ages):.2f}")
    print(f"Gender - Male: {sum(genders)}, Female: {len(genders) - sum(genders)}")
    print(f"\nList of failed images saved to {failed_file}")

if __name__ == "__main__":
    # Configuration
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2'
    output_file = 'age_gender_features.json'
    
    # Extract and save metadata
    extract_and_save_metadata(base_path, output_file)


# Total unique images to process: 12589
# Processing images: 100%|███████████████████████████████████████████████████████████████████████████████████████| 12589/12589 [05:23<00:00, 38.90it/s]

# Successfully processed 12589 images
# Failed to process 0 images
# Saving results to age_gender_features.json

# Dataset Statistics:
# Age - Mean: 38.81, Std: 17.52
# Gender - Male: 5749, Female: 6840

# List of failed images saved to age_gender_features_failed.txt