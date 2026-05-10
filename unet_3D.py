# unet_3D.py

import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            # Upsampling using transposed convolution
            self.conv1 = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            # Downsampling or standard convolution
            self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu  = nn.SiLU()

    def forward(self, x, t):
        # First convolution
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Inject time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 3] # Reshape to [Batch, Channels, 1, 1, 1]
        h = h + time_emb

        # Second convolution
        h = self.bnorm2(self.relu(self.conv2(h)))
        return h

class UNet3D(nn.Module):
    def __init__(self, in_channels=7, out_channels=7, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Multi-Layer Perceptron to process the plasma parameters [Ma, Ms]:
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, time_emb_dim), # 2 inputs: Ma and Ms
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial projection
        self.conv0 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        
        # Downsampling (Encoder)
        self.down1 = Block3D(32, 64, time_emb_dim)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = Block3D(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = Block3D(128, 128, time_emb_dim)
        
        # Upsampling (Decoder)
        # We separate the Up-Conv from the standard Block to fix the dimension mismatch
        self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.up1 = Block3D(128 + 128, 64, time_emb_dim) # 128 from upconv, 128 from x2
        
        self.upconv2 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.up2 = Block3D(64 + 64, 32, time_emb_dim)   # 64 from upconv, 64 from x1
        
        # Final output projection
        self.final_conv = nn.Conv3d(32 + 32, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep, cond):
        # Combine the time embedding with the physical parameters embedding:
        t = self.time_mlp(timestep) + self.cond_mlp(cond)
        
        # Initial
        x0 = self.conv0(x)
        
        # Encoder
        x1 = self.down1(x0, t)        # Shape: (B, 64, 64, 64, 64)
        p1 = self.pool1(x1)           # Shape: (B, 64, 32, 32, 32)
        
        x2 = self.down2(p1, t)        # Shape: (B, 128, 32, 32, 32)
        p2 = self.pool2(x2)           # Shape: (B, 128, 16, 16, 16)
        
        # Bottleneck
        bn = self.bottleneck(p2, t)   # Shape: (B, 128, 16, 16, 16)
        
        # Decoder
        # 1. Upsample bn from 16->32
        up_1 = self.upconv1(bn)       # Shape: (B, 128, 32, 32, 32)
        # 2. Concatenate with x2 (size 32) and pass through block
        u1 = self.up1(torch.cat([up_1, x2], dim=1), t) # Shape: (B, 64, 32, 32, 32)
        
        # 1. Upsample u1 from 32->64
        up_2 = self.upconv2(u1)       # Shape: (B, 64, 64, 64, 64)
        # 2. Concatenate with x1 (size 64) and pass through block
        u2 = self.up2(torch.cat([up_2, x1], dim=1), t) # Shape: (B, 32, 64, 64, 64)
        
        # Final concatenation with x0 to preserve ultra-fine detail
        out = self.final_conv(torch.cat([u2, x0], dim=1)) # Shape: (B, 7, 64, 64, 64)
        return out

# --- Testing the U-Net ---
if __name__ == "__main__":
    # Initialize the model
    model = UNet3D()

    # Create a dummy batch identical to what your DataLoader outputs
    # Shape: [Batch_Size, Channels, Z, Y, X]
    dummy_x = torch.randn(4, 7, 64, 64, 64)

    # Create random time steps for each item in the batch
    # Shape: [Batch_Size]
    dummy_t = torch.randint(0, 1000, (4,))

    print("Feeding data forward through the 3D U-Net...")
    output = model(dummy_x, dummy_t)

    print(f"Input Shape:  {dummy_x.shape}")
    print(f"Output Shape: {output.shape}")

    if dummy_x.shape == output.shape:
        print("Success! The U-Net preserves the spatial dimensions and channel counts.")

