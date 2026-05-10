import h5py
import numpy as np

# Path to one of the downloaded files
file_path = 'data/train/MHD_Ma_0.7_Ms_0.7.hdf5'

with h5py.File(file_path, 'r') as f:
    print("--- HDF5 File Metadata ---")
    print(f"Keys in the file: {list(f.keys())}")
    
    # We expect to see keys like 'density', 'velocity', 'mag_field' or similar
    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(f"\nKey: {key}")
            print(f"  Shape: {f[key].shape}")
            print(f"  Dtype: {f[key].dtype}")
            
            # Let's see a small statistical sample of the data
            sample = f[key][0] # Get the first snapshot
            print(f"  Min: {np.min(sample):.4f}, Max: {np.max(sample):.4f}, Mean: {np.mean(sample):.4f}")

    # Check for any global attributes (physics parameters)
    print("\n--- Global Attributes ---")
    for attr in f.attrs:
        print(f"{attr}: {f.attrs[attr]}")