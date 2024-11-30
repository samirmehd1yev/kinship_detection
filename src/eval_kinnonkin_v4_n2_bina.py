import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from kinship_model_v4_newdata import KinshipVerificationModel, Config

class KinshipVerifier:
    def __init__(self, model_path, threshold_path=None):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = KinshipVerificationModel(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load threshold
        if threshold_path and Path(threshold_path).exists():
            with open(threshold_path, 'r') as f:
                threshold_info = json.load(f)
            self.threshold = threshold_info['optimal_threshold']
        else:
            self.threshold = 1.05  # Default threshold if not provided
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            return img.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {str(e)}")
    
    def get_embedding(self, image):
        """Get embedding for a single image"""
        with torch.no_grad():
            embedding = self.model(image.to(self.device))
        return embedding
    
    def verify(self, image1_path, image2_path):
        """Verify kinship between two images"""
        # Preprocess images
        img1 = self.preprocess_image(image1_path)
        img2 = self.preprocess_image(image2_path)
        
        # Get embeddings
        with torch.no_grad():
            embed1 = self.get_embedding(img1)
            embed2 = self.get_embedding(img2)
            
            # Calculate distance
            distance = F.pairwise_distance(embed1, embed2).item()
            
            # Calculate confidence score (sigmoid scaled)
            confidence = float(torch.sigmoid(10 * (1.1 - torch.tensor(distance))))
            
            # Make prediction
            is_kin = distance < self.threshold
            
            result = {
                'prediction': 'KIN' if is_kin else 'NON-KIN',
                'distance': distance,
                'confidence': confidence,
                'threshold_used': self.threshold
            }
            
            return result

def main():
    # Example usage
    model_path = '/path/to/best_model.pth'
    threshold_path = '/path/to/threshold_info.json'
    
    # Initialize verifier
    verifier = KinshipVerifier(model_path, threshold_path)
    
    # Example verification
    image1_path = 'path/to/image1.jpg'
    image2_path = 'path/to/image2.jpg'
    
    try:
        result = verifier.verify(image1_path, image2_path)
        print("\nKinship Verification Results:")
        print("-" * 50)
        print(f"Prediction: {result['prediction']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Threshold: {result['threshold_used']:.4f}")
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")

if __name__ == "__main__":
    main()