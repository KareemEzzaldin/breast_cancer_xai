import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HybridResNetViT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, num_heads=8, depth=4):
        super(HybridResNetViT, self).__init__()
        
        # 1. ResNet-50 Backbone (Local Features Extractor)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Stop before pooling and linear layers to keep spatial dimension 7x7
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. Projection Layer
        # ResNet outputs 2048 channels. We compress it to 'embed_dim' (e.g., 768) for the Transformer
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        
        # 3. Vision Transformer (ViT) Components
        num_patches = 7 * 7  # Because ResNet output spatial size is 7x7
        
        # Class token (learnable parameter)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional Embedding (adds location info to the patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Classification Head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # -- Step 1: ResNet Extraction --
        # Output shape: [Batch, 2048, 7, 7]
        features = self.feature_extractor(x)
        
        # -- Step 2: Projection & Flattening --
        # Output shape: [Batch, 768, 7, 7]
        x = self.proj(features)
        
        # Flatten spatial dims: [Batch, 768, 49]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        
        # Transpose to sequence format: [Batch, 49, 768] (Batch, Sequence Length, Embedding Dim)
        x = x.transpose(1, 2)
        
        # -- Step 3: ViT Processing --
        # Add CLS token: [Batch, 50, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embeddings
        x = x + self.pos_embed
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # -- Step 4: Classification --
        # Extract the output of the CLS token only (index 0)
        cls_output = x[:, 0, :]
        
        # Final prediction: [Batch, Num_Classes]
        logits = self.head(cls_output)
        
        return logits

def main():
    print("Initializing Full Hybrid Architecture (ResNet-50 + ViT)...")
    model = HybridResNetViT(num_classes=2)
    
    # Create a dummy tensor (Batch Size=2, Channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("Testing full forward pass...")
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Final output shape (Logits): {output.shape}")
    print("Hybrid Model is fully built and ready for training!")

if __name__ == "__main__":
    main()