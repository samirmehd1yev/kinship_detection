# With this script you can test the kinship model with your own images
# command: python src/test_models.py --image1 <path_to_image1> --image2 <path_to_image2> --save_viz

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import onnx
from onnx2torch import convert
import argparse
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import warnings
from insightface.utils import face_align
import os
warnings.filterwarnings('ignore')

# Define relationship types and mappings
RELATIONSHIP_TYPES = ['ms', 'md', 'fs', 'fd', 'ss', 'bb', 'sibs']
REL_TO_IDX = {rel: idx for idx, rel in enumerate(RELATIONSHIP_TYPES)}
IDX_TO_REL = {idx: rel for rel, idx in REL_TO_IDX.items()}

# Gender-based relationship constraints
VALID_RELATIONSHIPS = {
    (0, 0): ['md', 'ss'],     # Female-Female
    (1, 1): ['fs', 'bb'],     # Male-Male
    (0, 1): ['ms', 'fd', 'sibs'],  # Female-Male
    (1, 0): ['ms', 'fd', 'sibs']   # Male-Female
}

RELATIONSHIP_DESCRIPTIONS = {
    'ms': 'Mother-Son',
    'md': 'Mother-Daughter',
    'fs': 'Father-Son',
    'fd': 'Father-Daughter',
    'ss': 'Sister-Sister',
    'bb': 'Brother-Brother',
    'sibs': 'Siblings (Brother-Sister)'
}

def preprocess_face(face_img):
    """Preprocess aligned face for the model."""
    if isinstance(face_img, np.ndarray):
        # Convert to float32 and normalize to match model requirements
        face_img = face_img.astype(np.float32)
        face_img = (face_img - 127.5) / 127.5
        # Convert to torch tensor and add batch dimension
        face_tensor = torch.from_numpy(face_img.transpose(2, 0, 1))
    return face_tensor

class FaceProcessor:
    def __init__(self, det_size=(128, 128)):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=det_size)
    
    def get_largest_face(self, faces):
        if not faces:
            return None
        return max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
    
    def process_image(self, img_path):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Detect faces
        faces = self.app.get(img)
        if not faces:
            raise ValueError(f"No faces detected in image: {img_path}")
        
        # Get largest face
        face = self.get_largest_face(faces)
        
        # Get gender (0: female, 1: male)
        gender = int(face.gender)
        print(f"gender: {gender}")
        
        # Get aligned face
        kps = face.kps
        aligned_face = face_align.norm_crop(img, kps)
        
        # Preprocess aligned face
        processed_face = preprocess_face(aligned_face)
        
        return {
            'aligned_face': processed_face,
            'gender': gender,
            'bbox': face.bbox,
            'kps': face.kps,
            'det_score': face.det_score,
            'raw_aligned': aligned_face  # Keep raw aligned face for visualization
        }

class RelationshipClassifier(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        self.backbone = convert(onnx.load(onnx_path))
        self.backbone.requires_grad_(False)
        
        self.embedding_dim = 512
        self.gender_dim = 2
        self.num_classes = len(RELATIONSHIP_TYPES)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2 + self.gender_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.ModuleDict({
                'layer': nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            }) for _ in range(2)
        ])
        
        # Final layers
        self.final_proj = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Linear(256, self.num_classes)
    
    def forward_one(self, x):
        emb = self.backbone(x)
        return emb
    
    def forward(self, x1, x2, gender_features):
        emb1 = F.normalize(self.forward_one(x1), p=2, dim=1)
        emb2 = F.normalize(self.forward_one(x2), p=2, dim=1)
        
        combined = torch.cat([emb1, emb2, gender_features], dim=1)
        features = self.input_proj(combined)
        
        for block in self.residual_blocks:
            residual = features
            features = block['layer'](features)
            features = features + residual
        
        features = self.final_proj(features)
        logits = self.classifier(features)
        
        # Apply gender-based masking
        gender1, gender2 = gender_features[:, 0], gender_features[:, 1]
        mask = torch.full_like(logits, float('-inf'))
        
        for i in range(len(gender1)):
            valid_rels = VALID_RELATIONSHIPS[(int(gender1[i].item()), int(gender2[i].item()))]
            valid_indices = [REL_TO_IDX[rel] for rel in valid_rels]
            mask[i, valid_indices] = 0
        
        return logits + mask

class KinshipVerificationModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        self.backbone = convert(onnx.load(onnx_path))
        self.backbone.requires_grad_(False)
    
    def forward_one(self, x):
        return self.backbone(x)
    
    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

def test_relationship(model, face1_data, face2_data, device):
    model.eval()
    
    with torch.no_grad():
        # Get preprocessed face tensors
        emb1 = face1_data['aligned_face'].unsqueeze(0).to(device)
        emb2 = face2_data['aligned_face'].unsqueeze(0).to(device)
        
        # Prepare gender features
        gender_features = torch.tensor([[face1_data['gender'], face2_data['gender']]], 
                                     dtype=torch.float32).to(device)
        
        # Get predictions
        logits = model(emb1, emb2, gender_features)
        probs = F.softmax(logits, dim=1)
        
        # Get valid relationships based on gender
        valid_rels = VALID_RELATIONSHIPS[(int(face1_data['gender']), int(face2_data['gender']))]
        valid_indices = [REL_TO_IDX[rel] for rel in valid_rels]
        
        # Filter predictions to only valid relationships
        valid_probs = probs[0, valid_indices]
        pred_idx = valid_indices[valid_probs.argmax().item()]
        pred_rel = IDX_TO_REL[pred_idx]
        confidence = probs[0, pred_idx].item()
        
        # Get top-3 valid predictions
        valid_rel_probs = [(IDX_TO_REL[valid_indices[i]], valid_probs[i].item()) 
                          for i in range(len(valid_indices))]
        top_3 = sorted(valid_rel_probs, key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'predicted_relationship': pred_rel,
            'relationship_description': RELATIONSHIP_DESCRIPTIONS[pred_rel],
            'confidence': confidence,
            'top_3_predictions': [
                {
                    'relationship': rel,
                    'description': RELATIONSHIP_DESCRIPTIONS[rel],
                    'confidence': conf
                } for rel, conf in top_3
            ]
        }

def test_kinship(model, face1_data, face2_data, threshold, device):
    model.eval()
    
    with torch.no_grad():
        # Get preprocessed face tensors
        emb1 = face1_data['aligned_face'].unsqueeze(0).to(device)
        emb2 = face2_data['aligned_face'].unsqueeze(0).to(device)
        
        # Get embeddings
        emb1, emb2 = model(emb1, emb2)
        
        # Calculate similarity
        similarity = F.cosine_similarity(
            F.normalize(emb1, p=2, dim=1),
            F.normalize(emb2, p=2, dim=1)
        ).item()
        
        # Normalize similarity score to confidence range
        # Map the similarity score from [-1, 1] to [0, 1] range
        normalized_similarity = (similarity + 1) / 2
        
        # Calculate kinship confidence
        # We use a scaled sigmoid function to emphasize the threshold region
        # and provide more interpretable confidence scores
        def sigmoid_scale(x, threshold, scale=10):
            x_scaled = scale * (x - threshold)
            return 1 / (1 + np.exp(-x_scaled))
        
        # Calculate confidence based on distance from threshold
        confidence = sigmoid_scale(similarity, threshold)
        
        # Determine kinship
        is_kin = similarity > threshold
        
        # Print debug info
        print(f"\nDebug Info:")
        print(f"Raw similarity score: {similarity:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Normalized similarity: {normalized_similarity:.4f}")
        print(f"Confidence score: {confidence:.4f}")
        
        return {
            'is_kin': bool(is_kin),
            'confidence': float(confidence),
            'similarity_score': float(similarity),
            'normalized_similarity': float(normalized_similarity)
        }
def draw_face_info(image_path, face_data, output_dir='.'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and draw on original image
    img = cv2.imread(image_path)
    bbox = face_data['bbox'].astype(int)
    kps = face_data['kps'].astype(int)
    
    # Draw bounding box
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Draw keypoints
    for x, y in kps:
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
    
    # Add gender label
    gender_text = "Female" if face_data['gender'] == 0 else "Male"
    cv2.putText(img, f"Gender: {gender_text}", (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save annotated image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    viz_path = os.path.join(output_dir, f'{base_name}_detected.jpg')
    cv2.imwrite(viz_path, img)
    
    # Save aligned face
    aligned_path = os.path.join(output_dir, f'{base_name}_aligned.jpg')
    cv2.imwrite(aligned_path, face_data['raw_aligned'])
    
    return viz_path, aligned_path

def main():
    parser = argparse.ArgumentParser(description='Test kinship models with your own images')
    parser.add_argument('--image1', required=True, help='Path to first image')
    parser.add_argument('--image2', required=True, help='Path to second image')
    parser.add_argument('--save_viz', action='store_true', help='Save visualization of face detection')
    parser.add_argument('--output_dir', default='results', help='Directory to save results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize face processor
    print("\nInitializing face processor...")
    face_processor = FaceProcessor()

    # Process images
    print("\nProcessing faces...")
    try:
        face1_data = face_processor.process_image(args.image1)
        face2_data = face_processor.process_image(args.image2)
        
        print(f"\nImage 1 - Gender: {'Female' if face1_data['gender'] == 0 else 'Male'}")
        print(f"Image 2 - Gender: {'Female' if face2_data['gender'] == 0 else 'Male'}")
        
        if args.save_viz:
            viz_path1, aligned_path1 = draw_face_info(args.image1, face1_data, args.output_dir)
            viz_path2, aligned_path2 = draw_face_info(args.image2, face2_data, args.output_dir)
            print(f"\nSaved visualizations for image 1:")
            print(f"- Detected face: {viz_path1}")
            print(f"- Aligned face: {aligned_path1}")
            print(f"\nSaved visualizations for image 2:")
            print(f"- Detected face: {viz_path2}")
            print(f"- Aligned face: {aligned_path2}")
    
    except Exception as e:
        print(f"Error processing faces: {str(e)}")
        return

    # Load models
    onnx_path = os.path.join(os.path.expanduser('~'), '.insightface/models/buffalo_l/w600k_r50.onnx')
    
    print("\nLoading relationship classifier...")
    rel_classifier = RelationshipClassifier(onnx_path)
    rel_checkpoint = torch.load('checkpoints/kin_relationship_v1/best_model.pth', 
                              map_location=device)
    rel_classifier.load_state_dict(rel_checkpoint['model_state_dict'])
    rel_classifier.to(device)
    
    print("Loading kinship verification model...")
    kin_verifier = KinshipVerificationModel(onnx_path)
    kin_checkpoint = torch.load('checkpoints/kin_binary_v1/best_model_l.pth',
                               map_location=device)
    kin_verifier.load_state_dict(kin_checkpoint['model_state_dict'])
    kin_verifier.to(device)
    
    # Test relationship classification
    print("\nTesting relationship classification...")
    rel_results = test_relationship(
        rel_classifier, 
        face1_data,
        face2_data,
        device
    )
    
    print(f"\nPredicted Relationship: {rel_results['relationship_description']}")
    print(f"Confidence: {rel_results['confidence']:.2%}")
    print("\nTop 3 Predictions:")
    for i, pred in enumerate(rel_results['top_3_predictions'], 1):
        print(f"{i}. {pred['description']} ({pred['confidence']:.2%})")
    
    # Test kinship verification
    print("\nTesting kinship verification...")
    kin_results = test_kinship(
        kin_verifier,
        face1_data,
        face2_data,
        kin_checkpoint['best_threshold'],
        device
    )
    
    print(f"\nKinship Verification Results:")
    print(f"Are they kin? {'Yes' if kin_results['is_kin'] else 'No'}")
    print(f"Confidence: {kin_results['confidence']:.2%}")
    print(f"Similarity Score: {kin_results['similarity_score']:.4f}")
    
    # Save results to file
    results = {
        'image1': {
            'path': args.image1,
            'gender': 'Female' if face1_data['gender'] == 0 else 'Male'
        },
        'image2': {
            'path': args.image2,
            'gender': 'Female' if face2_data['gender'] == 0 else 'Male'
        },
        'relationship_prediction': {
            'predicted_relationship': rel_results['relationship_description'],
            'confidence': float(rel_results['confidence']),
            'top_3_predictions': [
                {
                    'relationship': pred['description'],
                    'confidence': float(pred['confidence'])
                } for pred in rel_results['top_3_predictions']
            ]
        },
        'kinship_verification': {
            'is_kin': kin_results['is_kin'],
            'confidence': float(kin_results['confidence']),
            'similarity_score': float(kin_results['similarity_score'])
        }
    }
    
    # Save results to JSON file
    results_path = os.path.join(args.output_dir, 'results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    main()