import math
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Dictionary to store the activations
activations = {}

# Forward hook function to capture activations
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Load pretrained AlexNet and register hooks for conv layers
model = models.alexnet(pretrained=True)
model.eval()

# Register hooks for conv5 layer (features index: 10)
model.features[10].register_forward_hook(get_activation('conv5'))

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    # Convert BGR to RGB and to PIL image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return transform(pil_img).unsqueeze(0)

def normalize_feature_map(fmap):
    fmap = fmap.numpy()
    fmap -= fmap.min()
    fmap /= (fmap.max() + 1e-5)
    return fmap

def build_activation_grid(act, available_width, available_height):
    """
    Creates a grid image from activation maps with frames around each feature map.
    Sizes the grid to fill the available space optimally.
    """
    num_channels = act.shape[0]
    
    # Calculate optimal grid dimensions to fill available space
    aspect_ratio = available_width / available_height
    cols = int(math.ceil(math.sqrt(num_channels * aspect_ratio)))
    rows = int(math.ceil(num_channels / cols))
    
    # Calculate maximum possible size for each cell while maintaining aspect ratio
    cell_width = (available_width // cols)
    cell_height = (available_height // rows)
    
    # Add padding and frame
    padding = 2  # Pixels of padding between cells
    frame_thickness = 1  # Thickness of white frame
    
    # Calculate actual map size accounting for padding and frame
    map_size = (
        cell_width - 2 * (padding + frame_thickness),
        cell_height - 2 * (padding + frame_thickness)
    )
    
    # Create empty grid
    grid_image = np.zeros((available_height, available_width, 3), dtype='uint8')
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < num_channels:
                # Calculate cell position
                y_start = r * cell_height
                x_start = c * cell_width
                
                # Get and process feature map
                fmap = act[idx]
                fmap = normalize_feature_map(fmap) * 255
                fmap = fmap.astype('uint8')
                fmap_resized = cv2.resize(fmap, map_size)
                fmap_color = cv2.applyColorMap(fmap_resized, cv2.COLORMAP_JET)
                
                # Add white frame
                cell = np.zeros((cell_height, cell_width, 3), dtype='uint8')
                cell.fill(255)  # White frame
                
                # Place feature map in center of cell
                y_offset = padding + frame_thickness
                x_offset = padding + frame_thickness
                cell[y_offset:y_offset+map_size[1], 
                     x_offset:x_offset+map_size[0]] = fmap_color
                
                # Place cell in grid
                grid_image[y_start:y_start+cell_height, 
                          x_start:x_start+cell_width] = cell
                
            idx += 1
    
    return grid_image

def live_visualization():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Create a named window and set it to fullscreen
    window_name = 'Neural Network Visualization'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Get screen dimensions
    screen_width = 1920  # Default full HD width
    screen_height = 1080  # Default full HD height
    
    print("Press 'q' to exit, 'f' to toggle fullscreen.")
    
    fullscreen = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess and pass frame through model
        input_image = preprocess_frame(frame)
        with torch.no_grad():
            _ = model(input_image)

        if 'conv5' in activations:
            act = activations['conv5'][0]
            
            # Define webcam size (smaller, fixed size in top-left corner)
            webcam_width = screen_width // 6  # 1/6th of screen width
            webcam_height = int(webcam_width * (frame.shape[0] / frame.shape[1]))
            
            # Resize webcam frame
            frame_small = cv2.resize(frame, (webcam_width, webcam_height))
            
            # Create main composite image
            composite = np.zeros((screen_height, screen_width, 3), dtype='uint8')
            
            # Place webcam in top-left corner
            composite[0:webcam_height, 0:webcam_width] = frame_small
            
            # Calculate space available for activation grid
            # Grid will fill the space to the right of webcam and below it
            grid_image = build_activation_grid(
                act,
                available_width=screen_width,
                available_height=screen_height
            )
            
            # Overlay grid image, skipping the webcam area
            mask = np.zeros((screen_height, screen_width), dtype=bool)
            mask[0:webcam_height, 0:webcam_width] = True
            composite[~mask] = grid_image[~mask]
            
            cv2.imshow(window_name, composite)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    live_visualization()