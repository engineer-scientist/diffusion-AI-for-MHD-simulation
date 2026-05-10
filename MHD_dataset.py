# MHD_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob
import os

class MHDDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.file_paths = glob.glob(os.path.join(data_dir, '*.hdf5'))
        self.samples = []
        
        print(f"Scanning {len(self.file_paths)} files in {data_dir}...")
        
        for fp in self.file_paths:
            basename = os.path.basename(fp)
            parts = basename.replace('.hdf5', '').split('_')
            try:
                Ma = float(parts[2])
                Ms = float(parts[4])
            except ValueError:
                continue 
            
            # Dynamically check how many trajectories this specific file has
            with h5py.File(fp, 'r') as f:
                num_traj = f['t0_fields']['density'].shape[0]
                num_steps = f['t0_fields']['density'].shape[1]
            
            # Index exact number of available snapshots
            for i in range(num_traj * num_steps):
                # We store num_steps so __getitem__ knows how to do the math
                self.samples.append((fp, i, Ma, Ms, num_steps))
                
        print(f"Total 3D snapshots indexed: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, local_idx, Ma, Ms, num_steps = self.samples[idx]
        
        # Safely calculate indices based on the specific file's time steps
        traj_idx = local_idx // num_steps
        step_idx = local_idx % num_steps
        
        with h5py.File(fp, 'r') as f:
            rho = torch.tensor(f['t0_fields']['density'][traj_idx, step_idx], dtype=torch.float32).unsqueeze(0)
            vel = torch.tensor(f['t1_fields']['velocity'][traj_idx, step_idx], dtype=torch.float32).permute(3, 0, 1, 2)
            mag = torch.tensor(f['t1_fields']['magnetic_field'][traj_idx, step_idx], dtype=torch.float32).permute(3, 0, 1, 2)
        
        x = torch.cat([rho, vel, mag], dim=0)
        cond = torch.tensor([Ma, Ms], dtype=torch.float32)
        
        return x, cond
