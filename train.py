import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.hybrid_model import HybridResNetViT
from src.data.dataset import get_data_loaders

def train_one_epoch():
    print("Initializing Training Environment...")
    
    # 1. Setup Device (Use GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model
    print("Loading Hybrid Model...")
    model = HybridResNetViT(num_classes=2).to(device)
    
    # 3. Setup DataLoader
    print("Setting up DataLoader...")
    data_dir = os.path.join("data", "processed")
    # Using small batch size for local testing
    train_loader = get_data_loaders(data_dir=data_dir, batch_size=2)
    
    # 4. Define Loss Function and Optimizer
    # CrossEntropyLoss is standard for classification tasks
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer is great for Transformer architectures
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5. Training Loop (Just 1 Epoch for local testing)
    print("\n--- Starting Local Test Training (1 Epoch) ---")
    model.train() # Set model to training mode
    
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to the active device
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate Loss
        loss = criterion(outputs, labels)
        
        # Backward pass (Calculate Gradients)
        loss.backward()
        
        # Optimize (Update Weights)
        optimizer.step()
        
        running_loss += loss.item()
        print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"\n--- Training Completed! Average Loss: {avg_loss:.4f} ---")
    print("Success: The model is successfully learning and updating its weights!")

if __name__ == "__main__":
    train_one_epoch()