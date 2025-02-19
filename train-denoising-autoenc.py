import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random

class NoiseGenerator:
    @staticmethod
    def gaussian_noise(image, mean=0, std=0.2):
        """Add colored Gaussian noise"""
        noise = torch.randn_like(image) * std + mean
        # Generate random color scaling for each channel
        color_scale = torch.rand(3, 1, 1) * 0.5 + 0.5  # Random values between 0.5 and 1
        noise = noise * color_scale
        return noise

    @staticmethod
    def salt_and_pepper(image, prob=0.1):
        """Add salt and pepper noise"""
        noise = torch.zeros_like(image)
        # Salt
        salt_mask = torch.rand_like(image) < (prob/2)
        # Pepper
        pepper_mask = torch.rand_like(image) < (prob/2)
        
        noise[salt_mask] = 1
        noise[pepper_mask] = 0
        
        return noise - image

    @staticmethod
    def speckle(image, intensity=0.2):
        """Add multiplicative speckle noise"""
        noise = torch.randn_like(image) * intensity
        return image * (1 + noise)

    @staticmethod
    def poisson(image, scale=10.0):
        """Add Poisson noise"""
        noise = torch.poisson(torch.ones_like(image) * scale) / scale - 1
        return noise

    @staticmethod
    def structured_noise(image, num_structures=5, max_size=20):
        """Add structured noise like lines, dots, or shapes"""
        noise = torch.zeros_like(image)
        for _ in range(num_structures):
            # Random position
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[2] - 1)
            
            # Random color
            color = torch.rand(3) * 0.5  # Random color with intensity up to 0.5
            
            # Random structure type
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

class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, image_size=112, noise_params=None):  # Changed default size to 112
        self.image_dir = image_dir
        self.image_size = image_size
        self.noise_params = noise_params or {
            'gaussian': {'prob': 1.0, 'mean': 0, 'std': 0.2},
            'salt_and_pepper': {'prob': 0.3, 'amount': 0.1},
            'speckle': {'prob': 0.3, 'intensity': 0.2},
            'poisson': {'prob': 0.3, 'scale': 10.0},
            'structured': {'prob': 0.3, 'num_structures': 5, 'max_size': 20}
        }
        
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Updated transform to ensure consistent size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        self.noise_generator = NoiseGenerator()
    
    def add_noise(self, image):
        noisy_image = image.clone()
        total_noise = torch.zeros_like(image)
        
        # Apply each type of noise based on its probability
        if random.random() < self.noise_params['gaussian']['prob']:
            total_noise += self.noise_generator.gaussian_noise(
                image, 
                self.noise_params['gaussian']['mean'],
                self.noise_params['gaussian']['std']
            )
        
        if random.random() < self.noise_params['salt_and_pepper']['prob']:
            total_noise += self.noise_generator.salt_and_pepper(
                image,
                self.noise_params['salt_and_pepper']['amount']
            )
        
        if random.random() < self.noise_params['speckle']['prob']:
            total_noise += self.noise_generator.speckle(
                image,
                self.noise_params['speckle']['intensity']
            )
        
        if random.random() < self.noise_params['poisson']['prob']:
            total_noise += self.noise_generator.poisson(
                image,
                self.noise_params['poisson']['scale']
            )
        
        if random.random() < self.noise_params['structured']['prob']:
            total_noise += self.noise_generator.structured_noise(
                image,
                self.noise_params['structured']['num_structures'],
                self.noise_params['structured']['max_size']
            )
        
        noisy_image += total_noise
        return torch.clamp(noisy_image, 0., 1.)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        noisy_image = self.add_noise(image)
        return noisy_image, image


class DenoisingAutoencoder(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 112x112x3 -> 56x56x64
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 2: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 3: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_rate),
            
            # Layer 4: 14x14x256 -> 7x7x512
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: 7x7x512 -> 14x14x256
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Layer 2: 14x14x256 -> 28x28x128
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Layer 3: 28x28x128 -> 56x56x64
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Layer 4: 56x56x64 -> 112x112x3
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # No need for size cropping as dimensions are correct
    
def train_denoising_autoencoder(model, train_loader, save_dir, num_epochs=100, device='cuda', save_interval=10):
    """
    Train the denoising autoencoder with continuous best model saving
    
    Args:
        model: The model to train
        train_loader: DataLoader containing the training data
        save_dir: Directory to save model checkpoints
        num_epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        save_interval: How often to save regular checkpoints (in epochs)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    model = model.to(device)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (noisy_images, clean_images) in enumerate(train_loader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy_images)
            
            # Verify shapes match
            assert outputs.shape == clean_images.shape, f"Shape mismatch: outputs {outputs.shape} vs targets {clean_images.shape}"
            
            # Combined loss
            mse_loss = mse_criterion(outputs, clean_images)
            l1_loss = l1_criterion(outputs, clean_images)
            loss = 0.8 * mse_loss + 0.2 * l1_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
        
        scheduler.step(avg_loss)
        
        # Save checkpoint if it's the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f'New best model saved with loss: {best_loss:.6f}')
        
        # Regular interval saving
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1}')

def main():
    # Parameters
    image_dir = "data/apple/train"
    save_dir = "checkpoints/denoising"  # Directory for saving model checkpoints
    image_size = 112
    batch_size = 800
    num_epochs = 1000
    save_interval = 10  # Save checkpoints every 10 epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Custom noise parameters
    noise_params = {
        'gaussian': {'prob': 1.0, 'mean': 0, 'std': 0.3},
        'salt_and_pepper': {'prob': 0.4, 'amount': 0.15},
        'speckle': {'prob': 0.4, 'intensity': 0.25},
        'poisson': {'prob': 0.4, 'scale': 15.0},
        'structured': {'prob': 0.4, 'num_structures': 8, 'max_size': 25}
    }
    
    # Create dataset and dataloader with consistent image size
    dataset = NoisyImageDataset(image_dir, image_size=image_size, noise_params=noise_params)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = DenoisingAutoencoder()
    
    # Added shape verification
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0]
    print(f"Input shape: {sample_input.shape}")
    sample_output = model(sample_input)
    print(f"Output shape: {sample_output.shape}")
    assert sample_input.shape == sample_output.shape, "Model output shape doesn't match input shape"
    
    # Train model with specified save directory
    train_denoising_autoencoder(
        model, 
        train_loader, 
        save_dir=save_dir,
        num_epochs=num_epochs, 
        device=device,
        save_interval=save_interval
    )

if __name__ == "__main__":
    main()