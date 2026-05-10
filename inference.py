# inference.py

import torch
import numpy as np
from unet_3D import UNet3D

def generate_plasma_state():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for generation")
    
    # 1. Load the trained model
    model = UNet3D().to(device)
    model.load_state_dict(torch.load('mhd_diffusion_unet_v1.pth', map_location=device, weights_only=True))
    model.eval() # Set to evaluation mode
    
    # 2. Diffusion Schedule (Must match training exactly)
    timesteps = 1000
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 3. Initialize pure noise (Shape: Batch=1, Channels=7, Z=64, Y=64, X=64)
    print("Initializing pure 3D noise...")
    x = torch.randn(1, 7, 64, 64, 64).to(device)
    
    # 4. Reverse Diffusion Loop (DDPM Algorithm)
    print("Starting reverse diffusion process (Denoising)...")
    with torch.no_grad(): # No gradients needed for generation
        for i in reversed(range(timesteps)):
            t = torch.tensor([i], device=device)
            
            # Predict the noise for the current step
            predicted_noise = model(x, t)
            
            # Fetch constants for step 'i'
            alpha_t = alpha[t].view(-1, 1, 1, 1, 1)
            alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1, 1)
            beta_t = beta[t].view(-1, 1, 1, 1, 1)
            
            # Add Langevin noise (except at the very last step t=0)
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            # DDPM Update Equation
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
            
            # Print progress every 100 steps
            if i % 100 == 0:
                print(f"Step {i:4d}/{timesteps}: Denoising...")
                
    print("Generation complete!")
    
    # 5. Save the generated tensor for visualization
    generated_mhd = x.cpu().numpy()
    np.save('generated_mhd_state.npy', generated_mhd)
    print("Saved generated plasma state to 'generated_mhd_state.npy'.")

if __name__ == "__main__":
    generate_plasma_state()
