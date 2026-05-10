import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from unet_3D import UNet3D

def generate_sample(model, device, Ma, Ms, timesteps=1000):
    print(f"Generating plasma state for Ma={Ma}, Ms={Ms}...")
    model.eval()
    
    # DDPM parameters
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 1. Start with pure 3D Gaussian noise (1 batch, 7 channels, 64x64x64 grid)
    x = torch.randn(1, 7, 64, 64, 64).to(device)
    cond = torch.tensor([[Ma, Ms]], dtype=torch.float32).to(device)
    
    # 2. Reverse Diffusion Loop
    with torch.no_grad():
        for t in reversed(range(timesteps)):
            if t % 100 == 0:
                print(f"Denoising step {t}/{timesteps}...")
                
            t_tensor = torch.tensor([t], dtype=torch.long).to(device)
            
            # Predict the noise
            predicted_noise = model(x, t_tensor, cond)
            
            # DDPM reverse step math
            alpha_t = alpha[t]
            a_bar_t = alpha_bar[t]
            beta_t = beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            # Subtract predicted noise and add variance
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - a_bar_t)) * predicted_noise)
            x = x + torch.sqrt(beta_t) * noise
            
    print("Generation complete!")
    return x.squeeze(0).cpu().numpy() # Return as (7, 64, 64, 64) numpy array

def plot_plasma(plasma_state, Ma, Ms):
    # Extract the mid-plane slice (Z = 32)
    density = plasma_state[0, :, :, 32]
    Bx = plasma_state[4, :, :, 32]
    By = plasma_state[5, :, :, 32]
    
    # Set up the grid for streamlines
    Y, X = np.mgrid[0:64, 0:64]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"AI-Generated MHD State | Mach: {Ma} | Sonic Mach: {Ms}", fontsize=16, fontweight='bold')
    
    # Plot Density
    im1 = ax1.imshow(density, cmap='magma', origin='lower')
    ax1.set_title("Plasma Density", fontsize=14)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot Magnetic Streamlines
    ax2.streamplot(X, Y, Bx, By, color='teal', density=1.5, linewidth=1)
    ax2.set_title("Magnetic Field Streamlines (Div(B) ≈ 0)", fontsize=14)
    ax2.set_xlim(0, 63)
    ax2.set_ylim(0, 63)
    
    plt.tight_layout()
    filename = f"plasma_Ma_{Ma}_Ms_{Ms}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MHD Plasma from trained AI")
    parser.add_argument("--ma", type=float, default=0.7, help="Mach Number")
    parser.add_argument("--ms", type=float, default=2.0, help="Sonic Mach Number")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained conditional model
    model = UNet3D().to(device)
    weights_path = 'MHD_diffusion_scaled_30_epochs.pth'
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Successfully loaded 30-epoch weights.")
    else:
        print(f"Error: Could not find {weights_path}. Check the file name.")
        exit(1)
        
    # Generate and plot
    state = generate_sample(model, device, args.ma, args.ms)
    plot_plasma(state, args.ma, args.ms)