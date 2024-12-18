import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import pickle
import time
from pathlib import Path

def process_batch(image_paths, app, existing_keypoints=None):
    """Process a batch of images"""
    batch_keypoints = {}
    batch_errors = []
    
    for img_path in image_paths:
        if not isinstance(img_path, str):
            continue
            
        if existing_keypoints and img_path in existing_keypoints:
            batch_keypoints[img_path] = existing_keypoints[img_path]
            continue
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            faces = app.get(img)
            if not faces:
                raise ValueError(f"No face detected in: {img_path}")
            
            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            kps = face.kps
            
            batch_keypoints[img_path] = kps
            
        except Exception as e:
            batch_errors.append((img_path, str(e)))
            print(f"\nError processing {img_path}: {str(e)}")
    
    return batch_keypoints, batch_errors

def preprocess_keypoints(df, output_path, app, batch_size=10):
    """Extract and save keypoints in batches"""
    # Load existing keypoints if any
    keypoints_dict = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'rb') as f:
                keypoints_dict = pickle.load(f)
            print(f"Loaded {len(keypoints_dict)} existing keypoints")
        except:
            print("Could not load existing keypoints, starting fresh")
            keypoints_dict = {}
    
    all_errors = []
    unique_images = set()
    
    # Collect all unique image paths
    for col in ['Anchor', 'Positive', 'Negative']:
        unique_images.update(df[col].dropna().unique())
    
    # Convert to list and filter out non-strings and already processed images
    unique_images = [img for img in unique_images 
                    if isinstance(img, str) and img not in keypoints_dict]
    total_images = len(unique_images)
    
    print(f"Processing {total_images} new images")
    
    # Process in batches
    for i in tqdm(range(0, total_images, batch_size), desc='Processing batches'):
        batch_paths = unique_images[i:i + batch_size]
        batch_keypoints, batch_errors = process_batch(batch_paths, app, keypoints_dict)
        
        # Update keypoints dictionary
        keypoints_dict.update(batch_keypoints)
        all_errors.extend(batch_errors)
        
        # Save intermediate results every 1000 images
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total_images:
            # Backup existing file if it exists
            if os.path.exists(output_path):
                backup_path = output_path + '.bak'
                os.rename(output_path, backup_path)
            
            # Save new keypoints
            with open(output_path, 'wb') as f:
                pickle.dump(keypoints_dict, f)
            print(f"\nSaved intermediate results: {len(keypoints_dict)} keypoints processed")
            
            # Save current errors
            if all_errors:
                error_df = pd.DataFrame(all_errors, columns=['image_path', 'error'])
                error_df.to_csv(output_path.replace('.pkl', '_errors.csv'), index=False)
    
    return keypoints_dict, all_errors

def main():
    # Initialize InsightFace with CPU only
    import os
    # Set CUDA device ID to 1 (second GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Then initialize your FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(128, 128))
    
    # Base paths
    base_path = '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2'
    output_dir = os.path.join(base_path, 'keypoints')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    splits = {
        'train': 'train_triplets_enhanced.csv',
        'val': 'val_triplets_enhanced.csv',
        'test': 'test_triplets_enhanced.csv'
    }
    
    for split_name, filename in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Load DataFrame
        df = pd.read_csv(os.path.join(base_path, filename))
        print(f"Loaded {len(df)} rows from {filename}")
        
        # Update paths
        for col in ['Anchor', 'Positive', 'Negative']:
            df[col] = df[col].str.replace('../data',
                '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data',
                regex=False)
        
        # Process and save keypoints
        output_path = os.path.join(output_dir, f'{split_name}_keypoints.pkl')
        keypoints_dict, errors = preprocess_keypoints(df, output_path, app)
        
        print(f"\n{split_name} split completed:")
        print(f"Processed {len(keypoints_dict)} images")
        print(f"Encountered {len(errors)} errors")
        print(f"Saved keypoints to: {output_path}")
        
        # Save final error report
        if errors:
            error_path = output_path.replace('.pkl', '_errors.csv')
            error_df = pd.DataFrame(errors, columns=['image_path', 'error'])
            error_df.to_csv(error_path, index=False)
            print(f"Saved error report to: {error_path}")

if __name__ == "__main__":
    main()


# (kinship_venv_insightface) [mehdiyev@alvis1 src]$ python preprocess_kpss.py 
# Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}, 'CUDAExecutionProvider': {'prefer_nhwc': '0', 'enable_skip_layer_norm_strict_mode': '0', 'tunable_op_max_tuning_duration_ms': '0', 'use_ep_level_unified_stream': '0', 'tunable_op_enable': '0', 'enable_cuda_graph': '0', 'cudnn_conv_use_max_workspace': '1', 'do_copy_in_default_stream': '1', 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'gpu_external_empty_cache': '0', 'gpu_external_free': '0', 'tunable_op_tuning_enable': '0', 'cudnn_conv1d_pad_to_nc1d': '0', 'gpu_external_alloc': '0', 'arena_extend_strategy': 'kNextPowerOfTwo', 'has_user_compute_stream': '0', 'gpu_mem_limit': '18446744073709551615', 'device_id': '0'}}
# find model: /cephyr/users/mehdiyev/Alvis/.insightface/models/buffalo_l/1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0
# Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}, 'CUDAExecutionProvider': {'prefer_nhwc': '0', 'enable_skip_layer_norm_strict_mode': '0', 'tunable_op_max_tuning_duration_ms': '0', 'use_ep_level_unified_stream': '0', 'tunable_op_enable': '0', 'enable_cuda_graph': '0', 'cudnn_conv_use_max_workspace': '1', 'do_copy_in_default_stream': '1', 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'gpu_external_empty_cache': '0', 'gpu_external_free': '0', 'tunable_op_tuning_enable': '0', 'cudnn_conv1d_pad_to_nc1d': '0', 'gpu_external_alloc': '0', 'arena_extend_strategy': 'kNextPowerOfTwo', 'has_user_compute_stream': '0', 'gpu_mem_limit': '18446744073709551615', 'device_id': '0'}}
# find model: /cephyr/users/mehdiyev/Alvis/.insightface/models/buffalo_l/2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
# Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}, 'CUDAExecutionProvider': {'prefer_nhwc': '0', 'enable_skip_layer_norm_strict_mode': '0', 'tunable_op_max_tuning_duration_ms': '0', 'use_ep_level_unified_stream': '0', 'tunable_op_enable': '0', 'enable_cuda_graph': '0', 'cudnn_conv_use_max_workspace': '1', 'do_copy_in_default_stream': '1', 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'gpu_external_empty_cache': '0', 'gpu_external_free': '0', 'tunable_op_tuning_enable': '0', 'cudnn_conv1d_pad_to_nc1d': '0', 'gpu_external_alloc': '0', 'arena_extend_strategy': 'kNextPowerOfTwo', 'has_user_compute_stream': '0', 'gpu_mem_limit': '18446744073709551615', 'device_id': '0'}}
# find model: /cephyr/users/mehdiyev/Alvis/.insightface/models/buffalo_l/det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0
# Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}, 'CUDAExecutionProvider': {'prefer_nhwc': '0', 'enable_skip_layer_norm_strict_mode': '0', 'tunable_op_max_tuning_duration_ms': '0', 'use_ep_level_unified_stream': '0', 'tunable_op_enable': '0', 'enable_cuda_graph': '0', 'cudnn_conv_use_max_workspace': '1', 'do_copy_in_default_stream': '1', 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'gpu_external_empty_cache': '0', 'gpu_external_free': '0', 'tunable_op_tuning_enable': '0', 'cudnn_conv1d_pad_to_nc1d': '0', 'gpu_external_alloc': '0', 'arena_extend_strategy': 'kNextPowerOfTwo', 'has_user_compute_stream': '0', 'gpu_mem_limit': '18446744073709551615', 'device_id': '0'}}
# find model: /cephyr/users/mehdiyev/Alvis/.insightface/models/buffalo_l/genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0
# Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}, 'CUDAExecutionProvider': {'prefer_nhwc': '0', 'enable_skip_layer_norm_strict_mode': '0', 'tunable_op_max_tuning_duration_ms': '0', 'use_ep_level_unified_stream': '0', 'tunable_op_enable': '0', 'enable_cuda_graph': '0', 'cudnn_conv_use_max_workspace': '1', 'do_copy_in_default_stream': '1', 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'gpu_external_empty_cache': '0', 'gpu_external_free': '0', 'tunable_op_tuning_enable': '0', 'cudnn_conv1d_pad_to_nc1d': '0', 'gpu_external_alloc': '0', 'arena_extend_strategy': 'kNextPowerOfTwo', 'has_user_compute_stream': '0', 'gpu_mem_limit': '18446744073709551615', 'device_id': '0'}}
# find model: /cephyr/users/mehdiyev/Alvis/.insightface/models/buffalo_l/w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
# set det-size: (128, 128)

# Processing train split...
# Loaded 123285 rows from train_triplets_enhanced.csv
# Processing 12442 new images
# Processing batches:   0%|                                                                                      | 0/1245 [00:00<?, ?it/s]/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/kinship_venv_insightface/lib/python3.12/site-packages/insightface/utils/transform.py:68: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
# To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
#   P = np.linalg.lstsq(X_homo, Y)[0].T # Affine matrix. 3 x 4
# Processing batches:   8%|██████                                                                       | 99/1245 [00:27<04:49,  3.96it/s]
# Saved intermediate results: 1000 keypoints processed
# Processing batches:  16%|████████████▏                                                               | 199/1245 [00:53<04:33,  3.82it/s]
# Saved intermediate results: 2000 keypoints processed
# Processing batches:  24%|██████████████████▎                                                         | 299/1245 [01:19<04:00,  3.94it/s]
# Saved intermediate results: 3000 keypoints processed
# Processing batches:  32%|████████████████████████▎                                                   | 399/1245 [01:44<03:39,  3.86it/s]
# Saved intermediate results: 4000 keypoints processed
# Processing batches:  40%|██████████████████████████████▍                                             | 499/1245 [02:09<03:09,  3.94it/s]
# Saved intermediate results: 5000 keypoints processed
# Processing batches:  48%|████████████████████████████████████▌                                       | 599/1245 [02:34<02:40,  4.02it/s]
# Saved intermediate results: 6000 keypoints processed
# Processing batches:  56%|██████████████████████████████████████████▋                                 | 699/1245 [03:00<02:20,  3.90it/s]
# Saved intermediate results: 7000 keypoints processed
# Processing batches:  64%|████████████████████████████████████████████████▊                           | 799/1245 [03:25<01:49,  4.06it/s]
# Saved intermediate results: 8000 keypoints processed
# Processing batches:  72%|██████████████████████████████████████████████████████▉                     | 899/1245 [03:50<01:26,  4.01it/s]
# Saved intermediate results: 9000 keypoints processed
# Processing batches:  80%|████████████████████████████████████████████████████████████▉               | 999/1245 [04:15<01:01,  3.97it/s]
# Saved intermediate results: 10000 keypoints processed
# Processing batches:  88%|██████████████████████████████████████████████████████████████████▏        | 1099/1245 [04:41<00:36,  3.96it/s]
# Saved intermediate results: 11000 keypoints processed
# Processing batches:  96%|████████████████████████████████████████████████████████████████████████▏  | 1199/1245 [05:06<00:11,  3.86it/s]
# Saved intermediate results: 12000 keypoints processed
# Processing batches: 100%|██████████████████████████████████████████████████████████████████████████▉| 1244/1245 [05:18<00:00,  3.82it/s]
# Saved intermediate results: 12442 keypoints processed
# Processing batches: 100%|███████████████████████████████████████████████████████████████████████████| 1245/1245 [05:18<00:00,  3.91it/s]

# train split completed:
# Processed 12442 images
# Encountered 0 errors
# Saved keypoints to: /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2/keypoints/train_keypoints.pkl

# Processing val split...
# Loaded 26403 rows from val_triplets_enhanced.csv
# Processing 10042 new images
# Processing batches:   0%|                                                                                      | 0/1005 [00:00<?, ?it/s]/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/kinship_venv_insightface/lib/python3.12/site-packages/insightface/utils/transform.py:68: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
# To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
#   P = np.linalg.lstsq(X_homo, Y)[0].T # Affine matrix. 3 x 4
# Processing batches:  10%|███████▌                                                                     | 99/1005 [00:25<03:45,  4.02it/s]
# Saved intermediate results: 1000 keypoints processed
# Processing batches:  20%|███████████████                                                             | 199/1005 [00:50<03:19,  4.04it/s]
# Saved intermediate results: 2000 keypoints processed
# Processing batches:  30%|██████████████████████▌                                                     | 299/1005 [01:15<02:55,  4.03it/s]
# Saved intermediate results: 3000 keypoints processed
# Processing batches:  40%|██████████████████████████████▏                                             | 399/1005 [01:40<02:48,  3.59it/s]
# Saved intermediate results: 4000 keypoints processed
# Processing batches:  50%|█████████████████████████████████████▋                                      | 499/1005 [02:06<02:06,  4.01it/s]
# Saved intermediate results: 5000 keypoints processed
# Processing batches:  60%|█████████████████████████████████████████████▎                              | 599/1005 [02:31<01:43,  3.92it/s]
# Saved intermediate results: 6000 keypoints processed
# Processing batches:  70%|████████████████████████████████████████████████████▊                       | 699/1005 [02:56<01:15,  4.03it/s]
# Saved intermediate results: 7000 keypoints processed
# Processing batches:  80%|████████████████████████████████████████████████████████████▍               | 799/1005 [03:22<00:51,  4.03it/s]
# Saved intermediate results: 8000 keypoints processed
# Processing batches:  89%|███████████████████████████████████████████████████████████████████▉        | 899/1005 [03:47<00:26,  3.99it/s]
# Saved intermediate results: 9000 keypoints processed
# Processing batches:  99%|███████████████████████████████████████████████████████████████████████████▌| 999/1005 [04:12<00:01,  4.01it/s]
# Saved intermediate results: 10000 keypoints processed
# Processing batches: 100%|██████████████████████████████████████████████████████████████████████████▉| 1004/1005 [04:14<00:00,  3.83it/s]
# Saved intermediate results: 10042 keypoints processed
# Processing batches: 100%|███████████████████████████████████████████████████████████████████████████| 1005/1005 [04:14<00:00,  3.95it/s]

# val split completed:
# Processed 10042 images
# Encountered 0 errors
# Saved keypoints to: /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2/keypoints/val_keypoints.pkl

# Processing test split...
# Loaded 25996 rows from test_triplets_enhanced.csv
# Processing 10111 new images
# Processing batches:   0%|                                                                                      | 0/1012 [00:00<?, ?it/s]/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/kinship_venv_insightface/lib/python3.12/site-packages/insightface/utils/transform.py:68: FutureWarning: `rcond` parameter will change to the default of machine precision times ``max(M, N)`` where M and N are the input matrix dimensions.
# To use the future default and silence this warning we advise to pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
#   P = np.linalg.lstsq(X_homo, Y)[0].T # Affine matrix. 3 x 4
# Processing batches:  10%|███████▌                                                                     | 99/1012 [00:25<04:04,  3.74it/s]
# Saved intermediate results: 1000 keypoints processed
# Processing batches:  20%|██████████████▉                                                             | 199/1012 [00:51<03:24,  3.97it/s]
# Saved intermediate results: 2000 keypoints processed
# Processing batches:  30%|██████████████████████▍                                                     | 299/1012 [01:17<02:55,  4.07it/s]
# Saved intermediate results: 3000 keypoints processed
# Processing batches:  39%|█████████████████████████████▉                                              | 399/1012 [01:42<02:40,  3.82it/s]
# Saved intermediate results: 4000 keypoints processed
# Processing batches:  49%|█████████████████████████████████████▍                                      | 499/1012 [02:08<02:08,  3.99it/s]
# Saved intermediate results: 5000 keypoints processed
# Processing batches:  59%|████████████████████████████████████████████▉                               | 599/1012 [02:33<01:44,  3.96it/s]
# Saved intermediate results: 6000 keypoints processed
# Processing batches:  69%|████████████████████████████████████████████████████▍                       | 699/1012 [02:58<01:19,  3.96it/s]
# Saved intermediate results: 7000 keypoints processed
# Processing batches:  79%|████████████████████████████████████████████████████████████                | 799/1012 [03:24<00:59,  3.58it/s]
# Saved intermediate results: 8000 keypoints processed
# Processing batches:  89%|███████████████████████████████████████████████████████████████████▌        | 899/1012 [03:49<00:27,  4.04it/s]
# Saved intermediate results: 9000 keypoints processed
# Processing batches:  99%|███████████████████████████████████████████████████████████████████████████ | 999/1012 [04:15<00:03,  3.73it/s]
# Saved intermediate results: 10000 keypoints processed
# Processing batches: 100%|██████████████████████████████████████████████████████████████████████████▉| 1011/1012 [04:18<00:00,  3.91it/s]
# Saved intermediate results: 10111 keypoints processed
# Processing batches: 100%|███████████████████████████████████████████████████████████████████████████| 1012/1012 [04:18<00:00,  3.91it/s]

# test split completed:
# Processed 10111 images
# Encountered 0 errors
# Saved keypoints to: /mimer/NOBACKUP/groups/naiss2023-22-1358/samir_code/kinship_project/data/processed/fiw/train/splits_no_overlap_hand2/keypoints/test_keypoints.pkl
# (kinship_venv_insightface) [mehdiyev@alvis1 src]$ 