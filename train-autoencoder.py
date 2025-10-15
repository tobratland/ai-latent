import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image

# Parameters
train_data_dir = "data/VincentVanGogh/train"
val_data_dir = "data/VincentVanGogh/val"
batch_size = 300
num_epochs = 1000
initial_lr = 1e-3
save_every = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if not self.image_files:
            raise FileNotFoundError(f"No valid image files found in {root_dir}")
            
        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, image  # Return image as both input and target

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
    
def create_data_loaders(train_dir, val_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    
    train_dataset = ImageDataset(root_dir=train_dir, transform=transform)
    val_dataset = ImageDataset(root_dir=val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def train_model():
    print(f"Using device: {device}")
    print(f"Training directory: {train_data_dir}")
    print(f"Validation directory: {val_data_dir}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_data_dir, val_data_dir, batch_size)
    
    # Initialize model, criterion, optimizer and scheduler
    model = Autoencoder(dropout_rate=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                 patience=10, verbose=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data)
                val_loss += loss.item() * data.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every save_every epochs
        if epoch % save_every == 0:
            checkpoint_path = f"./model_checkpoints/checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_vangogh_model.pth")
            print("Saved new best model!")

if __name__ == "__main__":
    train_model()