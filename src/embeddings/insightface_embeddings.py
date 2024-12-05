from insightface.app import FaceAnalysis
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import pickle
from collections import defaultdict
import hashlib
import time


def validate_image_path(image_path):
    """Validate if image path exists and is readable"""
    if not os.path.exists(image_path):
        return False, f"File does not exist: {image_path}"
    if not os.path.isfile(image_path):
        return False, f"Not a file: {image_path}"
    if not os.access(image_path, os.R_OK):
        return False, f"File not readable: {image_path}"
    return True, None

def process_image(image_path, recognition_model=None, verbose=False):
    """Process single image and return embedding using InsightFace"""
    try:
        # Validate image path
        is_valid, error_msg = validate_image_path(image_path)
        if not is_valid:
            print(error_msg)
            return None

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}")
            return None

        if verbose:
            print(f"\nProcessing: {image_path}")
            print(f"Image shape: {img.shape}")
            print(f"Image dtype: {img.dtype}")

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to (112, 112) if necessary
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))

        # Get embedding using InsightFace
        embedding = recognition_model.get_feat(img).flatten()

        # Validate embedding
        if embedding is None or embedding.size == 0:
            print(f"Empty embedding generated for {image_path}")
            return None

        if verbose:
            print(f"Raw embedding shape: {embedding.shape}")
            print(f"Raw embedding range: [{embedding.min():.2f}, {embedding.max():.2f}]")

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            print(f"Warning: Very small norm {norm} for {image_path}")
            return None

        embedding = embedding / norm

        if verbose:
            print(f"Normalized embedding norm: {np.linalg.norm(embedding):.6f}")

        # Final validation
        if not np.all(np.isfinite(embedding)):
            print(f"Invalid values in embedding for {image_path}")
            return None

        return embedding

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def verify_saved_embeddings(embeddings_dict, output_dir):
    """Verify the saved embeddings can be loaded and are valid"""
    try:
        # Save embeddings
        embeddings_file = os.path.join(output_dir, 'arcface_embeddings.pkl')
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        
        # Try to load them back
        with open(embeddings_file, 'rb') as f:
            loaded_dict = pickle.load(f)
        
        # Verify contents
        if len(loaded_dict) != len(embeddings_dict):
            print("Warning: Number of embeddings changed after save/load")
            return False
            
        # Check a few random embeddings
        import random
        sample_keys = random.sample(list(embeddings_dict.keys()), min(5, len(embeddings_dict)))
        for key in sample_keys:
            if not np.allclose(embeddings_dict[key]['embedding'], loaded_dict[key]['embedding']):
                print(f"Warning: Embedding mismatch after save/load for {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error verifying saved embeddings: {str(e)}")
        return False

def generate_and_save_embeddings(csv_path, output_dir, sample_size=None, model_name="buffalo_l"):
    """Generate and save embeddings for all unique images using InsightFace"""
    
    start_time = time.time()
    
    # Validate inputs
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    
    # Load CSV with validation
    try:
        data = pd.read_csv(csv_path)
        required_columns = ['Anchor', 'Positive', 'Negative']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Update paths in the CSV
        data['Anchor'] = data['Anchor'].str.replace(
            '../data',
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
            regex=False
        )
        data['Positive'] = data['Positive'].str.replace(
            '../data',
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
            regex=False
        )
        data['Negative'] = data['Negative'].str.replace(
            '../data',
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
            regex=False
        )
    except Exception as e:
        raise Exception(f"Error loading CSV: {str(e)}")
        
    if sample_size:
        data = data.sample(n=min(sample_size, len(data)), random_state=42)
    print(f"Processing {len(data)} triplets")
    
    # Get unique image paths
    unique_images = set()
    for col in ['Anchor', 'Positive', 'Negative']:
        unique_images.update(data[col].unique())
    unique_images = list(unique_images)
    print(f"Found {len(unique_images)} unique images")
    
    # Initialize InsightFace model
    app = FaceAnalysis(
        name=model_name,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0)
    recognition_model = app.models.get('recognition')

    print(f"\nUsing InsightFace model: {recognition_model}")
    
    # Process images and save embeddings
    embeddings_dict = {}
    failed_images = []
    processing_times = []
    
    print("\nGenerating embeddings using InsightFace...")
    for img_path in tqdm(unique_images):
        process_start = time.time()
        
        # Create a shorter key for storage (just the filename)
        img_key = os.path.basename(img_path)
        
        # Skip if already processed (in case of duplicates)
        if img_key in embeddings_dict:
            continue
        
        # Generate embedding using InsightFace
        embedding = process_image(img_path, recognition_model=recognition_model, verbose=False)
        
        process_time = time.time() - process_start
        processing_times.append(process_time)
        
        if embedding is not None:
            # Additional validation of embedding
            if embedding.shape != (512,):  # Facenet512 should have 512-dimensional embeddings
                print(f"Warning: Unexpected embedding shape {embedding.shape} for {img_path}")
                failed_images.append(img_path)
                continue
                
            embeddings_dict[img_key] = {
                'full_path': img_path,
                'embedding': embedding,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'checksum': hashlib.md5(embedding.tobytes()).hexdigest()
            }
        else:
            failed_images.append(img_path)
    
    # Verify saved embeddings
    print("\nVerifying saved embeddings...")
    if not verify_saved_embeddings(embeddings_dict, output_dir):
        print("Warning: Embedding verification failed!")
    
    # Calculate and save statistics
    total_time = time.time() - start_time
    avg_time_per_image = np.mean(processing_times) if processing_times else 0
    
    stats = {
        'total_images': len(unique_images),
        'successful_embeddings': len(embeddings_dict),
        'failed_images': len(failed_images),
        'success_rate': len(embeddings_dict) / len(unique_images) * 100,
        'total_processing_time': total_time,
        'average_time_per_image': avg_time_per_image
    }
    
    # Save all results
    embeddings_file = os.path.join(output_dir, 'arcface_embeddings.pkl')
    failed_file = os.path.join(output_dir, 'failed_images.txt')
    stats_file = os.path.join(output_dir, 'embedding_stats.txt')
    
    # Save failed images list
    with open(failed_file, 'w') as f:
        for img_path in failed_images:
            f.write(f"{img_path}\n")
    
    # Save detailed statistics
    with open(stats_file, 'w') as f:
        f.write("Embedding Generation Statistics:\n")
        f.write(f"Total unique images: {stats['total_images']}\n")
        f.write(f"Successful embeddings: {stats['successful_embeddings']}\n")
        f.write(f"Failed images: {stats['failed_images']}\n")
        f.write(f"Success rate: {stats['success_rate']:.2f}%\n")
        f.write(f"Total processing time: {stats['total_processing_time']:.2f} seconds\n")
        f.write(f"Average time per image: {stats['average_time_per_image']:.2f} seconds\n")
        f.write("\nEmbedding file checksum: ")
        f.write(hashlib.md5(open(embeddings_file, 'rb').read()).hexdigest())
    
    print("\nEmbedding Generation Complete:")
    print(f"Successfully processed: {len(embeddings_dict)} images")
    print(f"Failed to process: {len(failed_images)} images")
    print(f"Success rate: {stats['success_rate']:.2f}%")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print(f"\nResults saved to: {output_dir}")
    print(f"Embeddings file size: {os.path.getsize(embeddings_file) / (1024*1024):.2f} MB")
    
    return embeddings_dict, failed_images, stats

def main():
    csv_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/hand_cleaned_filtered_triplets_with_labels.csv'
    output_dir = 'insightface_embeddings'

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        embeddings_dict, failed_images, stats = generate_and_save_embeddings(
            csv_path=csv_path,
            output_dir=output_dir,
            model_name="buffalo_l",
            sample_size=None  # Set to None to process all images
        )
        print("\nProcessing completed successfully.")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Using InsightFace model: <insightface.model_zoo.arcface_onnx.ArcFaceONNX object at 0x7fb9e8679df0>

# Generating embeddings using InsightFace...
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12615/12615 [01:32<00:00, 136.26it/s]

# Verifying saved embeddings...

# Embedding Generation Complete:
# Successfully processed: 12615 images
# Failed to process: 0 images
# Success rate: 100.00%
# Total processing time: 96.13 seconds
# Average time per image: 0.01 seconds

# Results saved to: insightface_embeddings
# Embeddings file size: 27.77 MB

# Processing completed successfully.
# (kinship_venv_insightface) [mehdiyev@alvis1 embeddings]$