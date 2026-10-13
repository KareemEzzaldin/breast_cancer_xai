import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.hybrid_model import HybridResNetViT
from torchvision import transforms
from PIL import Image

def generate_heatmap(image_path, model, device):
    """
    Generates a Grad-CAM heatmap for a given image using the hybrid model.
    """
    print(f"Processing image: {os.path.basename(image_path)}")
    
    # 1. Prepare the image exactly as the model expects it
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Open image, convert to RGB, and apply transforms
    original_img_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_img_pil).unsqueeze(0).to(device) # Add batch dimension
    
    # 2. Setup Grad-CAM
    # We target the last layer of our ResNet feature extractor (layer4)
    # This is where the spatial features (7x7) are most concentrated before going to ViT
    target_layers = [model.feature_extractor[-1]]
    
    # Initialize the CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 3. Generate the Heatmap
    # We don't pass specific targets, so it defaults to the highest scoring class
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :] # Extract the single image from the batch
    
    # 4. Visualization (Merge Heatmap with Original Image)
    # Prepare the original image for visualization (resize and scale to 0-1)
    img_cv = cv2.resize(np.array(original_img_pil), (224, 224))
    img_cv = np.float32(img_cv) / 255
    
    # Overlay the heatmap
    visualization = show_cam_on_image(img_cv, grayscale_cam, use_rgb=True)
    
    # 5. Display the result
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_cv)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("XAI Grad-CAM Heatmap")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Heatmap generated successfully!")

def main():
    print("Initializing XAI Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load our Hybrid Model
    # Note: We are using it un-trained for now, just to test the XAI pipeline.
    # The heatmap might look random now, but will make sense after Kaggle training.
    model = HybridResNetViT(num_classes=2).to(device)
    model.eval() # Set to evaluation mode
    
    # Pick the first processed image to test
    processed_dir = os.path.join("data", "processed")
    test_images = [f for f in os.listdir(processed_dir) if f.endswith('.png')]
    
    if not test_images:
        print("Error: No images found in data/processed/")
        return
        
    test_image_path = os.path.join(processed_dir, test_images[0])
    
    generate_heatmap(test_image_path, model, device)

if __name__ == "__main__":
    main()