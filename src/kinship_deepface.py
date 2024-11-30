from deepface import DeepFace
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import pickle

class EmbeddingExtractor:
    def __init__(self, cache_dir='embeddings_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_embedding(self, img_path):
        """Get embedding with caching"""
        # Create cache key from image path
        cache_key = os.path.basename(img_path).replace('.jpg', '.pkl')
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Check if embedding is cached
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        try:
            # Convert relative path to absolute
            abs_path = os.path.abspath(img_path.replace("../data", "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data"))
            
            # Get embedding
            embedding_objs = DeepFace.represent(
                img_path=abs_path,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="retinaface",
                align=True
            )
            embedding = np.array(embedding_objs[0]["embedding"])
            
            # Cache the embedding
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            return embedding
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None

    def preprocess_dataset(self, csv_path, output_path=None):
        """Preprocess entire dataset and save embeddings"""
        df = pd.read_csv(csv_path)
        print(f"\nExtracting embeddings from {len(df)} triplets...")
        
        embeddings = []
        labels = []
        
        # Process all unique images first
        unique_images = set(df['Anchor'].tolist() + df['Positive'].tolist() + df['Negative'].tolist())
        
        print("Processing unique images...")
        for img_path in tqdm(unique_images):
            _ = self.get_embedding(img_path)  # This will cache the embedding
        
        print("\nCreating pairs...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Get cached embeddings
            anchor_emb = self.get_embedding(row['Anchor'])
            positive_emb = self.get_embedding(row['Positive'])
            negative_emb = self.get_embedding(row['Negative'])
            
            if anchor_emb is None or positive_emb is None or negative_emb is None:
                continue
            
            # Store positive pair (kin)
            embeddings.append(np.concatenate([anchor_emb, positive_emb]))
            labels.append(1)
            
            # Store negative pair (non-kin)
            embeddings.append(np.concatenate([anchor_emb, negative_emb]))
            labels.append(0)
        
        data = {
            'embeddings': np.array(embeddings),
            'labels': np.array(labels)
        }
        
        # Save processed data if output path is provided
        if output_path:
            print(f"\nSaving processed data to {output_path}")
            np.savez_compressed(output_path, **data)
        
        return data

def build_model(input_dim):
    """Create kinship detection model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on GPUs")

    # Now check GPU availability
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPU Devices: ", tf.config.list_physical_devices('GPU'))
    print("Is built with CUDA:", tf.test.is_built_with_cuda())
    
    # Test if GPU is being used with a simple operation
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("Tensor operation result device:", c.device)
        print("Tensor operation result:", c)
    
    # Original GPU memory growth code
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled on GPUs")
    
    # Parameters
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Create embedding extractor
    extractor = EmbeddingExtractor('embeddings_cache')
    
    # Process datasets
    print("Processing datasets...")
    
    # Check if processed data exists
    if not all(os.path.exists(f) for f in ['train_data.npz', 'val_data.npz', 'test_data.npz']):
        print("Preprocessing datasets...")
        train_data = extractor.preprocess_dataset(
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/train_triplets_enhanced.csv',
            'train_data.npz'
        )
        val_data = extractor.preprocess_dataset(
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/val_triplets_enhanced.csv',
            'val_data.npz'
        )
        test_data = extractor.preprocess_dataset(
            '/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_data/processed/fiw/train/splits_no_overlap_hand/test_triplets_enhanced.csv',
            'test_data.npz'
        )
    else:
        print("Loading preprocessed data...")
        train_data = dict(np.load('train_data.npz'))
        val_data = dict(np.load('val_data.npz'))
        test_data = dict(np.load('test_data.npz'))
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data['embeddings'].astype(np.float32), train_data['labels'].astype(np.float32))
    ).shuffle().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_data['embeddings'].astype(np.float32), val_data['labels'].astype(np.float32))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data['embeddings'].astype(np.float32), test_data['labels'].astype(np.float32))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Create and compile model
    input_dim = train_data['embeddings'].shape[1]
    model = build_model(input_dim)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'best_kinship_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs'
            )
        ]
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_dataset)
    for name, value in zip(model.metrics_names, test_results):
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()