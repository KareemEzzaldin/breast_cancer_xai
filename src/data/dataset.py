import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BreastCancerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the processed images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Grab all .png images from the processed folder
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir}")

    def __len__(self):
        # Returns the total number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Get image path
        img_path = self.image_paths[idx]
        
        # 2. Open image using PIL
        # Convert to RGB because ResNet expects 3 channels, even though medical images are grayscale
        image = Image.open(img_path).convert('RGB')
        
        # 3. Dummy Labeling for testing (In real scenario, extract label from CSV or filename)
        # For now, let's randomly assign 0 (Benign) or 1 (Malignant) based on even/odd index
        label = 1 if idx % 2 == 0 else 0 
        
        # 4. Apply transforms (resize, tensor, normalization)
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_data_loaders(data_dir, batch_size=2):
    """
    Creates and returns the DataLoader.
    """
    # Standard transforms for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Standard ImageNet normalization (Expected by pre-trained ResNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Instantiate the Dataset
    dataset = BreastCancerDataset(data_dir=data_dir, transform=transform)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def main():
    print("Testing DataLoader pipeline...")
    
    # Path to our processed images
    processed_dir = os.path.join("data", "processed")
    
    # Get the DataLoader
    dataloader = get_data_loaders(data_dir=processed_dir, batch_size=2)
    
    # Fetch a single batch
    images, labels = next(iter(dataloader))
    
    print(f"Batch Images Shape: {images.shape} -> [Batch_Size, Channels, Height, Width]")
    print(f"Batch Labels: {labels}")
    print("DataLoader is successfully feeding data to PyTorch!")

if __name__ == "__main__":
    main()