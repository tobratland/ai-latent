import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

class Autoencoder(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 100x100x3 -> 50x50x32
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 2: 50x50x32 -> 25x25x64
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 3: 25x25x64 -> 13x13x64
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 4: 13x13x64 -> 7x7x32
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: 7x7x32 -> 13x13x64
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 2: 13x13x64 -> 25x25x64
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 3: 25x25x64 -> 50x50x32
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 4: 50x50x32 -> 100x100x3
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x[:, :, :100, :100]  # Ensure output size matches input
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def process_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    return input_tensor.cpu(), output_tensor.cpu()

def show_image_comparison(input_tensor, output_tensor, title="Original vs Reconstructed"):
    # Convert tensors to images
    input_img = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(input_img)
    plt.title('Original')
    plt.axis('off')
    
    # Show reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(output_img)
    plt.title('Reconstructed')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot
    #save_path = f"comparison_{os.path.basename(title)}.png"
    #plt.savefig(save_path)
    #print(f"Saved comparison to {save_path}")
    
    # Show the plot and wait for it to display
    plt.show()
    plt.close()

def process_directory(directory_path, model, device, max_images=10):
    image_files = [f for f in os.listdir(directory_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    print(f"Found {len(image_files)} images in directory")
    
    for i, image_file in enumerate(image_files[:max_images]):
        print(f"Processing image {i+1}/{min(max_images, len(image_files))}: {image_file}")
        image_path = os.path.join(directory_path, image_file)
        input_tensor, output_tensor = process_image(image_path, model, device)
        show_image_comparison(input_tensor, output_tensor, title=f"Image {i+1}: {image_file}")

def compute_reconstruction_error(input_tensor, output_tensor):
    """Compute pixel-wise reconstruction error"""
    # Convert to numpy arrays and reshape to (H,W,C)
    input_img = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # Compute absolute difference
    error = np.abs(input_img - output_img)
    
    # Average across color channels for visualization
    error_map = np.mean(error, axis=2)
    
    # Compute statistics
    mean_error = np.mean(error)
    max_error = np.max(error)
    per_channel_error = np.mean(error, axis=(0,1))
    
    return {
        'error_map': error_map,
        'mean_error': mean_error,
        'max_error': max_error,
        'channel_errors': per_channel_error
    }

def show_image_comparison(input_tensor, output_tensor, title="Original vs Reconstructed"):
    # Convert tensors to images
    input_img = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # Compute reconstruction error
    error_data = compute_reconstruction_error(input_tensor, output_tensor)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show original image
    ax1.imshow(input_img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Show reconstructed image
    ax2.imshow(output_img)
    ax2.set_title('Reconstructed')
    ax2.axis('off')
    
    # Show error heatmap
    error_plot = ax3.imshow(error_data['error_map'], cmap='hot')
    ax3.set_title('Reconstruction Error')
    ax3.axis('off')
    plt.colorbar(error_plot, ax=ax3)
    
    # Add error information
    error_text = f'Mean Error: {error_data["mean_error"]:.4f}\n'
    error_text += f'Max Error: {error_data["max_error"]:.4f}\n'
    error_text += f'Channel Errors (RGB): ({error_data["channel_errors"][0]:.4f}, '
    error_text += f'{error_data["channel_errors"][1]:.4f}, {error_data["channel_errors"][2]:.4f})'
    
    plt.figtext(0.5, 0.02, error_text, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot
    save_path = f"comparison_{os.path.basename(title)}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved comparison to {save_path}")
    
    # Show the plot and wait for it to display
    plt.show()
    plt.close()

def process_directory(directory_path, model, device, max_images=10):
    # Get and sort image files
    image_files = sorted([f for f in os.listdir(directory_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
    
    print("\nFound files (in order):")
    for idx, f in enumerate(image_files):
        print(f"{idx + 1}. {f}")
    
    print(f"\nTotal images found: {len(image_files)}")
    print(f"Will process up to {min(max_images, len(image_files))} images")
    
    # Store all errors for summary
    all_errors = []
    
    # Process each image
    for i, image_file in enumerate(image_files[:max_images]):
        # Build full path and verify file exists
        image_path = os.path.join(directory_path, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: File not found: {image_path}")
            continue
            
        print(f"\nProcessing image {i+1}/{min(max_images, len(image_files))}")
        print(f"File: {image_file}")
        print(f"Full path: {image_path}")
        
        # Load and process image
        try:
            input_tensor, output_tensor = process_image(image_path, model, device)
            
            # Verify tensors are different
            if torch.allclose(input_tensor, output_tensor):
                print("Warning: Input and output tensors are identical!")
            
            # Compute and store errors
            error_data = compute_reconstruction_error(input_tensor, output_tensor)
            error_data['filename'] = image_file
            all_errors.append(error_data)
            
            # Show comparison
            show_image_comparison(input_tensor, output_tensor, title=f"Image {i+1}: {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    if not all_errors:
        print("No images were successfully processed!")
        return
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    avg_error = np.mean([e['mean_error'] for e in all_errors])
    max_error = np.max([e['max_error'] for e in all_errors])
    print(f"Average reconstruction error across all images: {avg_error:.4f}")
    print(f"Maximum reconstruction error across all images: {max_error:.4f}")
    
    # Print individual image statistics
    print("\nPer-image Statistics:")
    print("-" * 50)
    for error in all_errors:
        print(f"File: {error['filename']}")
        print(f"  Mean Error: {error['mean_error']:.4f}")
        print(f"  Max Error: {error['max_error']:.4f}")
        print(f"  Channel Errors (RGB): {error['channel_errors'].tolist()}")
        print()

def main():
    # Set paths
    model_path = "best_vangogh_model.pth"
    image_path = "vangogh_val"
    
    # Verify paths
    print(f"Checking paths...")
    print(f"Model path: {os.path.abspath(model_path)}")
    print(f"Image path: {os.path.abspath(image_path)}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}")
        return
    
    # Load model
    print("\nLoading model...")
    model, device = load_model(model_path)
    print(f"Model loaded and running on {device}")
    
    # Process images
    if os.path.isdir(image_path):
        print(f"\nProcessing directory: {image_path}")
        process_directory(image_path, model, device)
    else:
        print(f"\nProcessing single image: {image_path}")
        try:
            input_tensor, output_tensor = process_image(image_path, model, device)
            show_image_comparison(input_tensor, output_tensor, 
                                title=os.path.basename(image_path))
        except Exception as e:
            print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    print("Starting inference...")
    main()
    print("\nDone!")