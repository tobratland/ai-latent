import gradio as gr
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import os

class SemanticImageSearch:
    def __init__(self, image_folder):
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("Loading and embedding images...")
        self.image_paths = []
        self.image_embeddings = []
        
        # Load all images from folder
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for ext in valid_extensions:
            self.image_paths.extend(Path(image_folder).glob(f'*{ext}'))
            self.image_paths.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        print(f"Found {len(self.image_paths)} images")
        
        # Pre-compute all image embeddings
        for img_path in self.image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize for cosine similarity
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                self.image_embeddings.append(image_features.cpu().numpy())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        self.image_embeddings = np.vstack(self.image_embeddings)
        print("Ready!")
    
    def search(self, query_text, top_k=3):
        if not query_text.strip():
            return []
        
        # Encode the text query
        inputs = self.processor(text=[query_text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        text_features = text_features.cpu().numpy()
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.image_embeddings, text_features.T).squeeze()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return images with their similarity scores
        results = []
        for idx in top_indices:
            results.append({
                'image': str(self.image_paths[idx]),
                'score': float(similarities[idx])
            })
        
        return results

# Initialize the search engine
IMAGE_FOLDER = "search_here"  # Change this to your image folder path
searcher = SemanticImageSearch(IMAGE_FOLDER)

def search_images(query_text):
    results = searcher.search(query_text, top_k=3)
    
    if not results:
        return None, None, None, "Enter a search query!"
    
    # Prepare images with scores
    images = []
    labels = []
    
    for i, result in enumerate(results):
        img = Image.open(result['image'])
        images.append(img)
        labels.append(f"Match {i+1} (Score: {result['score']:.3f})")
    
    # Pad with None if less than 3 results
    while len(images) < 3:
        images.append(None)
        labels.append("")
    
    status = f"Found {len(results)} matches for: '{query_text}'"
    
    return images[0], images[1], images[2], status

# Create Gradio interface
with gr.Blocks(title="Semantic Image Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Semantic Image Search with CLIP")
    gr.Markdown("Type any description and find matching images from the collection!")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Search Query", 
            placeholder="e.g., 'a red car', 'sunset over mountains', 'person smiling'",
            scale=4
        )
        search_btn = gr.Button("🔍 Search", scale=1, variant="primary")
    
    status_text = gr.Textbox(label="Status", interactive=False)
    
    with gr.Row():
        img1 = gr.Image(label="Top Match", type="pil", height=300)
        img2 = gr.Image(label="Second Match", type="pil", height=300)
        img3 = gr.Image(label="Third Match", type="pil", height=300)
    
    # Search on button click
    search_btn.click(
        fn=search_images,
        inputs=[query_input],
        outputs=[img1, img2, img3, status_text]
    )
    
    # Also search on Enter key
    query_input.submit(
        fn=search_images,
        inputs=[query_input],
        outputs=[img1, img2, img3, status_text]
    )
    
    gr.Markdown("""
    ### How it works:
    1. **CLIP** embeds both images and text into the same latent space
    2. When you search, your text is embedded into this space
    3. We find the images whose embeddings are **closest** to your text embedding
    4. No training needed - CLIP already understands semantic similarity!
    """)

if __name__ == "__main__":
    demo.launch(share=False)