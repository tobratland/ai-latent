import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from copy import deepcopy

class NoiseGenerator:
    @staticmethod
    def gaussian_noise(image, mean=0, std=0.2):
        noise = torch.randn_like(image) * std + mean
        color_scale = torch.rand(3, 1, 1) * 0.5 + 0.5
        noise = noise * color_scale
        return noise

    @staticmethod
    def salt_and_pepper(image, prob=0.1):
        noise = torch.zeros_like(image)
        salt_mask = torch.rand_like(image) < (prob/2)
        pepper_mask = torch.rand_like(image) < (prob/2)
        noise[salt_mask] = 1
        noise[pepper_mask] = 0
        return noise - image

    @staticmethod
    def speckle(image, intensity=0.2):
        noise = torch.randn_like(image) * intensity
        return image * (1 + noise)

    @staticmethod
    def structured_noise(image, num_structures=5, max_size=20):
        noise = torch.zeros_like(image)
        for _ in range(num_structures):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[2] - 1)
            color = torch.rand(3) * 0.5
            
            structure_type = random.choice(['dot', 'line', 'square'])
            
            if structure_type == 'dot':
                size = random.randint(1, 5)
                x_indices = torch.clamp(torch.arange(x-size, x+size), 0, image.shape[1]-1)
                y_indices = torch.clamp(torch.arange(y-size, y+size), 0, image.shape[2]-1)
                for i in x_indices:
                    for j in y_indices:
                        if (i-x)**2 + (j-y)**2 <= size**2:
                            noise[:, i, j] = color
            
            elif structure_type == 'line':
                length = random.randint(5, max_size)
                angle = random.random() * 2 * np.pi
                for i in range(length):
                    new_x = int(x + i * np.cos(angle))
                    new_y = int(y + i * np.sin(angle))
                    if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[2]:
                        noise[:, new_x, new_y] = color
            
            else:  # square
                size = random.randint(3, 10)
                x_indices = torch.clamp(torch.arange(x, x+size), 0, image.shape[1]-1)
                y_indices = torch.clamp(torch.arange(y, y+size), 0, image.shape[2]-1)
                for i in x_indices:
                    for j in y_indices:
                        noise[:, i, j] = color
        
        return noise

class DenoisingAutoencoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder().to(device)
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, device

def add_noise_to_image(image_tensor):
    noise_generator = NoiseGenerator()
    noisy_image = image_tensor.clone()
    total_noise = torch.zeros_like(image_tensor)
    
    # Add different types of noise
    total_noise += noise_generator.gaussian_noise(image_tensor, mean=0, std=0.3)
    total_noise += noise_generator.salt_and_pepper(image_tensor, prob=0.15)
    total_noise += noise_generator.speckle(image_tensor, intensity=0.25)
    
    noisy_image += total_noise
    return torch.clamp(noisy_image, 0., 1.)

def process_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Match the size used in training
        transforms.ToTensor()
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    clean_tensor = transform(image).unsqueeze(0)
    
    # Add noise
    noisy_tensor = add_noise_to_image(clean_tensor)
    
    # Move to device and process
    noisy_tensor = noisy_tensor.to(device)
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
    
    return clean_tensor.cpu(), noisy_tensor.cpu(), denoised_tensor.cpu()

def compute_reconstruction_error(input_tensor, output_tensor):
    input_img = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_img = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    error = np.abs(input_img - output_img)
    error_map = np.mean(error, axis=2)
    
    mean_error = np.mean(error)
    max_error = np.max(error)
    per_channel_error = np.mean(error, axis=(0,1))
    
    return {
        'error_map': error_map,
        'mean_error': mean_error,
        'max_error': max_error,
        'channel_errors': per_channel_error
    }

def show_image_comparison(clean_tensor, noisy_tensor, denoised_tensor, title="Denoising Results"):
    # Convert tensors to images
    clean_img = clean_tensor.squeeze(0).permute(1, 2, 0).numpy()
    noisy_img = noisy_tensor.squeeze(0).permute(1, 2, 0).numpy()
    denoised_img = denoised_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # Compute reconstruction error
    error_data = compute_reconstruction_error(clean_tensor, denoised_tensor)
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Show original image
    ax1.imshow(clean_img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Show noisy image
    ax2.imshow(noisy_img)
    ax2.set_title('Noisy')
    ax2.axis('off')
    
    # Show denoised image
    ax3.imshow(denoised_img)
    ax3.set_title('Denoised')
    ax3.axis('off')
    
    # Show error heatmap
    error_plot = ax4.imshow(error_data['error_map'], cmap='hot')
    ax4.set_title('Reconstruction Error')
    ax4.axis('off')
    plt.colorbar(error_plot, ax=ax4)
    
    # Add error information
    error_text = f'Mean Error: {error_data["mean_error"]:.4f}\n'
    error_text += f'Max Error: {error_data["max_error"]:.4f}\n'
    error_text += f'Channel Errors (RGB): ({error_data["channel_errors"][0]:.4f}, '
    error_text += f'{error_data["channel_errors"][1]:.4f}, {error_data["channel_errors"][2]:.4f})'
    
    plt.figtext(0.5, 0.02, error_text, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot
    #save_path = f"denoising_{os.path.basename(title)}.png"
    #plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #print(f"Saved comparison to {save_path}")
    
    plt.show()
    plt.close()

def process_directory(directory_path, model, device, max_images=10):
    image_files = sorted([f for f in os.listdir(directory_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
    
    print(f"\nFound {len(image_files)} images in directory")
    print("Files to process:")
    for idx, f in enumerate(image_files[:max_images]):
        print(f"{idx + 1}. {f}")
    
    all_errors = []
    
    for i, image_file in enumerate(image_files[:max_images]):
        print(f"\nProcessing image {i+1}/{min(max_images, len(image_files))}: {image_file}")
        image_path = os.path.join(directory_path, image_file)
        
        try:
            clean_tensor, noisy_tensor, denoised_tensor = process_image(image_path, model, device)
            
            # Compute and store errors
            error_data = compute_reconstruction_error(clean_tensor, denoised_tensor)
            error_data['filename'] = image_file
            all_errors.append(error_data)
            
            # Show comparison
            show_image_comparison(clean_tensor, noisy_tensor, denoised_tensor, 
                                title=f"Image {i+1}: {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    # Print summary statistics
    if all_errors:
        print("\nSummary Statistics:")
        print("-" * 50)
        avg_error = np.mean([e['mean_error'] for e in all_errors])
        max_error = np.max([e['max_error'] for e in all_errors])
        print(f"Average reconstruction error across all images: {avg_error:.4f}")
        print(f"Maximum reconstruction error across all images: {max_error:.4f}")
        
        print("\nPer-image Statistics:")
        for error in all_errors:
            print(f"\nFile: {error['filename']}")
            print(f"  Mean Error: {error['mean_error']:.4f}")
            print(f"  Max Error: {error['max_error']:.4f}")
            print(f"  Channel Errors (RGB): {error['channel_errors'].tolist()}")

def main():
    # Set paths
    model_path = "checkpoints/denoising/best_model.pth"  # Path to your trained model
    image_path = "denoise_apples_val"     # Path to your test images
    
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
        process_directory(image_path, model, device)
    else:
        try:
            clean_tensor, noisy_tensor, denoised_tensor = process_image(image_path, model, device)
            show_image_comparison(clean_tensor, noisy_tensor, denoised_tensor,
                                title=os.path.basename(image_path))
        except Exception as e:
            print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    print("Starting denoising inference...")
    main()
    print("\nDone!")