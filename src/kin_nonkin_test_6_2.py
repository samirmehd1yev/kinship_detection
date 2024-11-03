import os
import numpy as np
import torch
import cv2
from torch import nn
from torch.nn import functional as F
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.utils import face_align

# # Initialize InsightFace
# app = FaceAnalysis(providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(128, 128))

class KinshipConfig:
    def __init__(self):
        # Model architecture
        self.input_size = 112  # Input image size
        self.face_embedding_size = 512
        
        # Training settings
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 5e-4
        self.num_epochs = 25
        
        # Data settings
        self.train_path = '../data/processed/fiw/train/splits/train_triplets.csv'
        self.val_path = '../data/processed/fiw/train/splits/val_triplets.csv'
        self.test_path = '../data/processed/fiw/train/splits/test_triplets.csv'

# Model components - Blocks
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual Block with SE attention"""
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Feature Extractor Network
class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, config.face_embedding_size)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

        layers = []
        layers.append(ResidualBlock(in_c, out_c, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.normalize(x, p=2, dim=1)

# Kinship Verification Model
class KinshipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extractor (shared weights)
        self.feature_extractor = FeatureExtractor(config)
        
        # Fusion layers
        fusion_size = config.face_embedding_size * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, fusion_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_size // 2, fusion_size // 4),
            nn.ReLU(inplace=True)
        )
        
        # Kinship verification head
        hidden_size = fusion_size // 4
        self.kinship_verifier = nn.Linear(hidden_size, 1)

    def forward(self, anchor, other):
        # Extract features
        anchor_features = self.feature_extractor(anchor)
        other_features = self.feature_extractor(other)
        
        # Concatenate features
        pair_features = torch.cat([anchor_features, other_features], dim=1)
        
        # Fuse features
        fused_features = self.fusion(pair_features)
        
        # Get kinship score
        kinship_score = self.kinship_verifier(fused_features)
        
        return {
            'kinship_score': kinship_score.squeeze(),
            'anchor_features': anchor_features,
            'other_features': other_features
        }

# Custom image processing functions
class ImageProcessor:
    @staticmethod
    def read_image(path):
        """Read image using OpenCV and convert to RGB"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def resize_image(img, size):
        """Resize image keeping aspect ratio"""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h))
    
    @staticmethod
    def pad_image(img, size):
        """Pad image to square"""
        h, w = img.shape[:2]
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        return cv2.copyMakeBorder(
            img, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    
    @staticmethod
    def align_face(img):
        """Align face using InsightFace"""
        faces = app.get(img)
        if not faces:
            raise ValueError("No faces detected.")
        face = faces[0]
        kps = face.kps.astype(int)
        aligned_face = face_align.norm_crop(img, kps)
        if aligned_face is None:
            raise ValueError("Failed to align face.")
        return aligned_face
    
    @staticmethod
    def preprocess_image(img):
        """Normalize image to [-1, 1] range"""
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2
        return img
    
    @staticmethod
    def crop_face(img):
        """Detect face using Haar cascades and crop"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            raise ValueError("No faces detected.")
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    
    @staticmethod
    def process_face(img_path, target_size=112):
        """Complete face processing pipeline"""
        try:
            
            # Read image
            img = ImageProcessor.read_image(img_path)
            
            # Crop face
            # img = ImageProcessor.crop_face(img)
            
            
            # Resize keeping aspect ratio
            img = ImageProcessor.resize_image(img, target_size)
            
            # Align face
            # img = ImageProcessor.align_face(img)
                        
            # Pad to square
            img = ImageProcessor.pad_image(img, target_size)
            
            # Preprocess
            img = ImageProcessor.preprocess_image(img)
            
            # Convert to torch tensor
            img = torch.from_numpy(img.transpose(2, 0, 1))
            
            return img
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None

def predict_kinship(model, img1_path, img2_path, device):
    img1 = ImageProcessor.process_face(img1_path)
    img2 = ImageProcessor.process_face(img2_path)
    
    
    if img1 is None or img2 is None:
        print("Error processing images.")
        return
    #save images in temp/ folder for checking
    # Convert images back to numpy arrays for saving
    img1_np = img1.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize images from [-1, 1] to [0, 1]
    img1_np = (img1_np + 1) / 2
    img2_np = (img2_np + 1) / 2
    
    # Save images in temp/ folder for checking
    os.makedirs("temp", exist_ok=True)
    cv2.imwrite("temp/img1.jpg", (img1_np * 255).astype(np.uint8))
    cv2.imwrite("temp/img2.jpg", (img2_np * 255).astype(np.uint8))
    
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img1, img2)
        kinship_score = outputs['kinship_score']
        # print(outputs)
        kinship_prob = torch.sigmoid(kinship_score).item()

    print(f"\nKinship Probability: {kinship_prob:.4f}")
    if (kinship_prob > 0.5):
        print("The individuals are likely to be kin.")
    else:
        print("The individuals are unlikely to be kin.")

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model_path = "/mimer/NOBACKUP/groups/naiss2023-22-1358/samir_kinship_model/output_kin_nonkin_model/model/best_kin_nonkin_model.pth"
    config = KinshipConfig()
    model = KinshipModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Paths to your own images
    img1_path = "../data/processed/fiw/train/train-faces/F0190/MID4/P02037_face1.jpg"
    img2_path = "/cephyr/users/mehdiyev/Alvis/kinship_project/data/processed/fiw/train/train-faces/F0010/MID4/P00107_face1.jpg"

    # Predict kinship
    predict_kinship(model, img1_path, img2_path, device)
   

    