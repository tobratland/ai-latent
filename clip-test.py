import torch
import open_clip
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class CLIPClassifier:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        """Initialize the CLIP classifier with specified model."""
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
    def predict(self, image_path: str, class_names: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the class of an image using CLIP.
        
        Args:
            image_path: Path to the image file
            class_names: List of class names to predict from
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples containing (class_name, probability)
        """
        # Load and preprocess image
        image = PIL.Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Prepare text tokens
        text_tokens = self.tokenizer([f"a photo of a {name}" for name in class_names]).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get top k predictions
            values, indices = similarity[0].topk(top_k)
            
        return [(class_names[idx], val.item()) for val, idx in zip(values, indices)]
    
    def display_prediction(self, image_path: str, predictions: List[Tuple[str, float]]):
        """Display the image with its predictions."""
        # Load image
        image = PIL.Image.open(image_path).convert('RGB')
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Input Image')
        
        # Display predictions
        plt.subplot(1, 2, 2)
        classes, probs = zip(*predictions)
        y_pos = np.arange(len(classes))
        
        plt.barh(y_pos, probs)
        plt.yticks(y_pos, classes)
        plt.xlabel('Probability')
        plt.title('CLIP Predictions')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = CLIPClassifier()
    
    # Example classes
    
    image_path = "clip_test/dog.jpg"
    #image_path = "clip_test/cat.jpg"
    classes = ["dog", "cat", "bird", "fish", "hamster"]
    
    #image_path = "clip_test/alfredo.jpg"
    #image_path = "clip_test/carbonara.jpg"
    #image_path = "clip_test/bolognese.jpg"
    #classes = ["spaghetti carbonara", "spaghetti bolognese", "spaghetti alfredo", "spaghetti aglio e olio", "spaghetti puttanesca"]

    #classes = ["rocket taking off", "rocket landing", "rocket exploding", "rocket crashing"]
    #image_path = "clip_test/starship.jpg" 
    #image_path = "clip_test/starship-t.jpg" 


    
    predictions = classifier.predict(image_path, classes)
    
    # Display results
    classifier.display_prediction(image_path, predictions)