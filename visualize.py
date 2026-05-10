# visualize.py

import numpy as np
import matplotlib.pyplot as plt

def plot_mhd_slice():
    print("Loading generated plasma state...")
    # Shape is [1, 7, 64, 64, 64] -> [Batch, Channels, Z, Y, X]
    data = np.load('generated_mhd_state.npy')
    
    # We will slice right through the middle of the Z-axis
    z_slice = 32
    
    # Extract Density (Channel 0)
    density = data[0, 0, z_slice, :, :]
    
    # Extract Magnetic Field X and Y (Channels 4 and 5)
    Bx = data[0, 4, z_slice, :, :]
    By = data[0, 5, z_slice, :, :]
    
    # Calculate magnetic field magnitude for the background
    B_mag = np.sqrt(Bx**2 + By**2)
    
    # Create coordinate grids for the streamlines
    Y, X = np.mgrid[0:64, 0:64]
    
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Density Contour ---
    im1 = axes[0].imshow(density, origin='lower', cmap='viridis')
    axes[0].set_title(f'Generated Plasma Density (Z = {z_slice})', fontsize=14)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im1, ax=axes[0], label='Density')
    
    # --- Plot 2: Magnetic Field Streamlines ---
    # We plot the magnitude as the background and overlay the vector streamlines
    im2 = axes[1].imshow(B_mag, origin='lower', cmap='magma')
    axes[1].streamplot(X, Y, Bx, By, color='white', linewidth=1, density=1.5)
    axes[1].set_title(f'Generated B-Field Streamlines (Z = {z_slice})', fontsize=14)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    fig.colorbar(im2, ax=axes[1], label='|B| Magnitude')
    
    plt.tight_layout()
    save_path = 'mhd_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Visualization saved to {save_path}")

if __name__ == "__main__":
    plot_mhd_slice()
