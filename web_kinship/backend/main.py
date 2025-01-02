from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import numpy as np
from torchvision import transforms
import cv2
import onnx
from onnx2torch import convert
from typing import List, Dict, Any
import logging
import os
from pathlib import Path
import time
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

app = FastAPI(title="Kinship Verification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return F.normalize(emb, p=2, dim=1)
    
    def forward(self, x1, x2, gender_features):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        
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

class ModelManager:
    def __init__(self):
        self.kin_model = None
        self.rel_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = None
        self.threshold = 0.5
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_model(self):
        try:
            self.face_detector = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_detector.prepare(ctx_id=0, det_size=(128, 128))
            logger.info("Face detector initialized")

            # Load ONNX backbone
            home = str(Path.home())
            onnx_path = os.path.join(home, '.insightface/models/buffalo_l/w600k_r50.onnx')
            
            # Load kinship verification model
            self.kin_model = KinshipVerificationModel(onnx_path)
            kin_checkpoint = torch.load('checkpoints_web/best_model_l.pth', map_location=self.device)
            self.kin_model.load_state_dict(kin_checkpoint['model_state_dict'])
            self.threshold = kin_checkpoint.get('best_threshold', 0.5)
            self.kin_model.eval()
            
            # Load relationship classifier
            self.rel_model = RelationshipClassifier(onnx_path)
            rel_checkpoint = torch.load('checkpoints_web/best_model_rel.pth', map_location=self.device)
            self.rel_model.load_state_dict(rel_checkpoint['model_state_dict'])
            self.rel_model.eval()
            
            self.kin_model.to(self.device)
            self.rel_model.to(self.device)
            
            logger.info("Models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def process_image_and_get_keypoints(self, image_bytes: bytes):
        """Process image and return both the processed image and keypoints."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            
            # Detect faces
            faces = self.face_detector.get(img_rgb)
            if not faces:
                try:
                    # Add padding
                    img_rgb = cv2.copyMakeBorder(img_rgb, 100, 100, 100, 100, 
                                            cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    faces = self.face_detector.get(img_rgb)
                except Exception as e:
                    logger.error(f"Error detecting faces: {str(e)}")
                    raise HTTPException(status_code=400, detail="No face detected")
            
            # Get largest face
            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            
            # Get gender
            gender = int(face.gender)
            # print(f"Gender = {face.gender}, int = {gender} ")
            
            # Get keypoints
            kps = face.kps.astype(np.float32)
            
            # Align face
            aligned_face = face_align.norm_crop(img_rgb, kps)
            if aligned_face is None:
                raise ValueError("Failed to align face")
            
            # Convert to PIL and tensor
            face_pil = Image.fromarray(aligned_face)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            return {
                'tensor': face_tensor,
                'gender': gender,
                'bbox': face.bbox,
                'kps': kps,
                'det_score': face.det_score
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def calculate_kinship_confidence(self, similarity: float, threshold: float, scale: float = 10.0) -> float:
        """
        Calculate kinship confidence using a scaled sigmoid function.
        
        Args:
            similarity: Cosine similarity score (-1 to 1)
            threshold: Model's decision threshold
            scale: Scaling factor for steepness of sigmoid curve
            
        Returns:
            Confidence score between 0 and 1
        """
        # Shift and scale the similarity to center around the decision threshold
        x = scale * (similarity - threshold)
        # Apply sigmoid function
        confidence = 1.0 / (1.0 + np.exp(-x))
        return float(confidence)

    async def verify_pair(self, image1_bytes: bytes, image2_bytes: bytes) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            # Process both images
            face1_data = self.process_image_and_get_keypoints(image1_bytes)
            face2_data = self.process_image_and_get_keypoints(image2_bytes)
            
            # Get face tensors and gender
            img1_tensor = face1_data['tensor']
            img2_tensor = face2_data['tensor']
            gender1 = face1_data['gender']
            gender2 = face2_data['gender']
            
            gender_features = torch.tensor([[gender1, gender2]], 
                                        dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # Kinship verification
                emb1, emb2 = self.kin_model(img1_tensor, img2_tensor)
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(
                    F.normalize(emb1, p=2, dim=1),
                    F.normalize(emb2, p=2, dim=1)
                )
                
                sim_value = float(similarity.item())
                print(f"Similarity = {sim_value}")
                is_kin = bool(sim_value > self.threshold)
                
                # Calculate confidence using sigmoid normalization
                confidence = self.calculate_kinship_confidence(
                    similarity=sim_value,
                    threshold=self.threshold,
                    scale=10.0 # Steepness of sigmoid curve
                )
                
                # Get relationship prediction
                gender_pair = (int(gender1), int(gender2))
                
                if is_kin:
                    # Get relationship predictions using relationship classifier
                    logits = self.rel_model(img1_tensor, img2_tensor, gender_features)
                    probs = F.softmax(logits, dim=1)
                    
                    # Get valid relationships based on gender
                    valid_rels = VALID_RELATIONSHIPS[gender_pair]
                    valid_indices = [REL_TO_IDX[rel] for rel in valid_rels]
                    
                    # Filter predictions to only valid relationships
                    valid_probs = probs[0, valid_indices]
                    
                    # Get top 3 predictions
                    top_k = min(3, len(valid_indices))
                    top_values, top_indices = torch.topk(valid_probs, top_k)
                    
                    relationship_predictions = []
                    for idx, (conf_idx, conf_val) in enumerate(zip(top_indices, top_values)):
                        rel_type = valid_rels[conf_idx]
                        relationship_predictions.append({
                            'type': rel_type,
                            'description': RELATIONSHIP_DESCRIPTIONS[rel_type],
                            'confidence': float(conf_val)
                        })
                else:
                    relationship_predictions = []
                
                result = {
                    "isKin": is_kin,
                    "confidence": confidence,
                    "similarity": sim_value, # Raw cosine similarity score
                    "gender1": "Female" if gender1 == 0 else "Male",
                    "gender2": "Female" if gender2 == 0 else "Male",
                    "relationships": relationship_predictions,
                    "processingTime": time.time() - start_time
                }
                
                return result
                    
        except Exception as e:
            logger.error(f"Error in verification: {str(e)}")
            raise


# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialize model and face detector on startup."""
    try:
        model_manager.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize on startup: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Check if the service is healthy."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/verify-kinship/")
async def verify_kinship(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    Verify kinship between two face images.
    Returns kinship prediction with confidence score and relationship prediction.
    """
    try:
        start_time = time.time()
        
        # Read image files
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        
        # Verify kinship and predict relationship
        result = await model_manager.verify_pair(img1_bytes, img2_bytes)

        
        processing_time = time.time() - start_time
        
        return {
            **result,
            "processingTime": processing_time
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze-group/")
async def analyze_group(image: UploadFile = File(...)):
    """Analyze a group photo for kinship relationships."""
    try:
        start_time = time.time()
        img_bytes = await image.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_img = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = model_manager.face_detector.get(img_rgb)
        
        if len(faces) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"At least two faces are required. Found {len(faces)} faces."
            )
        
        # Process each face
        face_data = []
        face_locations = []
        
        # Draw faces on image
        vis_img = original_img.copy()
        
        for idx, face in enumerate(faces):
            try:
                bbox = face.bbox.astype(int)
                kps = face.kps.astype(np.float32)
                gender = int(face.gender)
                
                
                # Draw rectangle
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Add number label with gender
                label = f"{idx+1} ({'M' if gender == 1 else 'F'})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                
                text_x = bbox[0]
                text_y = bbox[1] - 10 if bbox[1] - 10 > text_size[1] else bbox[1] + 30
                
                cv2.rectangle(vis_img, 
                            (text_x, text_y - text_size[1] - 5),
                            (text_x + text_size[0], text_y + 5),
                            (255, 255, 255), -1)
                
                cv2.putText(vis_img, label, (text_x, text_y), 
                           font, font_scale, (0, 0, 255), font_thickness)
                
                # Process face
                aligned_face = face_align.norm_crop(img_rgb, kps)
                if aligned_face is not None:
                    face_pil = Image.fromarray(aligned_face)
                    face_tensor = model_manager.transform(face_pil)
                    face_tensor = face_tensor.unsqueeze(0).to(model_manager.device)
                    
                    face_data.append({
                        'tensor': face_tensor,
                        'gender': gender
                    })
                    
                    face_locations.append({
                        'bbox': bbox.tolist(),
                        'label': idx + 1,
                        'gender': 'Male' if gender == 1 else 'Female'
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process face {idx+1}: {str(e)}")
                continue
        
        # Convert visualization image to base64
        _, buffer = cv2.imencode('.jpg', vis_img)
        vis_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Compare all pairs
        results = []
        for i in range(len(face_data)):
            for j in range(i + 1, len(face_data)):
                try:
                    with torch.no_grad():
                        # Kinship verification
                        emb1, emb2 = model_manager.kin_model(
                            face_data[i]['tensor'],
                            face_data[j]['tensor']
                        )
                        
                        similarity = F.cosine_similarity(
                            F.normalize(emb1, p=2, dim=1),
                            F.normalize(emb2, p=2, dim=1)
                        ).item()
                        
                        # Only process if kinship is detected
                        if similarity >= model_manager.threshold:
                            # Get relationship prediction
                            gender_features = torch.tensor(
                                [[face_data[i]['gender'], face_data[j]['gender']]],
                                dtype=torch.float32
                            ).to(model_manager.device)
                            
                            logits = model_manager.rel_model(
                                face_data[i]['tensor'],
                                face_data[j]['tensor'],
                                gender_features
                            )
                            probs = F.softmax(logits, dim=1)
                            
                            # Get valid relationships
                            gender_pair = (face_data[i]['gender'], face_data[j]['gender'])
                            valid_rels = VALID_RELATIONSHIPS[gender_pair]
                            valid_indices = [REL_TO_IDX[rel] for rel in valid_rels]
                            
                            # Get top prediction
                            valid_probs = probs[0, valid_indices]
                            top_idx = valid_probs.argmax().item()
                            rel_type = valid_rels[top_idx]
                            
                            results.append({
                                "pair": [i+1, j+1],
                                "locations": [face_locations[i], face_locations[j]],
                                "similarity": float(similarity),
                                "confidence": float(abs(similarity - 0.5) * 2),
                                "relationship": {
                                    "type": rel_type,
                                    "description": RELATIONSHIP_DESCRIPTIONS[rel_type],
                                    "confidence": float(valid_probs[top_idx])
                                }
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to compare faces {i+1} and {j+1}: {str(e)}")
                    continue
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "totalFaces": len(faces),
            "processedFaces": len(face_data),
            "processingTime": processing_time,
            "face_locations": face_locations,
            "visualizedImage": f"data:image/jpeg;base64,{vis_img_base64}"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing group photo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )