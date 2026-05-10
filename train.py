# train.py

import os
import csv
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from MHD_dataset import MHDDataset
from unet_3D import UNet3D

def compute_divergence_loss(pred_x0):
    B = pred_x0[:, 4:7, :, :, :]
    dBx_dx = B[:, 0, :, :, 2:] - B[:, 0, :, :, :-2] 
    dBy_dy = B[:, 1, :, 2:, :] - B[:, 1, :, :-2, :] 
    dBz_dz = B[:, 2, 2:, :, :] - B[:, 2, :-2, :, :] 
    div_B = (dBx_dx[:, 1:-1, 1:-1, :] + dBy_dy[:, 1:-1, :, 1:-1] + dBz_dz[:, :, 1:-1, 1:-1]) / 2.0
    return torch.mean(div_B ** 2)

def train():
    # 1. Initialize Distributed Process Group
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if global_rank == 0:
        print("Initializing Distributed Training...")

    # 2. Data Loading with Distributed Sampler
    dataset = MHDDataset('data/train/')
    sampler = DistributedSampler(dataset)
    # 4 GPUs per node, batch size 2 per GPU = Effective batch size of 8 per node
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=4, pin_memory=True)
    
    # 3. Model & Optimizer Setup
    model = UNet3D().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision for speed
    
    # 4. Diffusion Schedule
    timesteps = 1000
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    epochs = 50 # Increased to 50 epochs
    
    # CSV Logging Setup (Only Master Node writes)
    if global_rank == 0:
        csv_file = open('training_metrics.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Avg_MSE', 'Avg_Div_Loss'])

    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch) # Crucial for shuffling in DDP
        epoch_mse = 0
        epoch_div = 0
        
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            current_batch_size = batch.shape[0]
            
            noise = torch.randn_like(batch).to(device)
            t = torch.randint(0, timesteps, (current_batch_size,)).to(device)
            a_bar_t = alpha_bar[t].view(-1, 1, 1, 1, 1)
            noisy_batch = torch.sqrt(a_bar_t) * batch + torch.sqrt(1 - a_bar_t) * noise
            
            optimizer.zero_grad()
            
            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast():
                predicted_noise = model(noisy_batch, t)
                mse_loss = F.mse_loss(predicted_noise, noise)
                pred_x0 = (noisy_batch - torch.sqrt(1 - a_bar_t) * predicted_noise) / torch.sqrt(a_bar_t)
                div_loss = compute_divergence_loss(pred_x0)
                total_loss = mse_loss + (0.1 * div_loss)
            
            # Backward pass with Scaler
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_mse += mse_loss.item()
            epoch_div += div_loss.item()
            
        # Logging
        if global_rank == 0:
            avg_mse = epoch_mse / len(dataloader)
            avg_div = epoch_div / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | Avg MSE: {avg_mse:.4f} | Avg Div(B): {avg_div:.4f}")
            csv_writer.writerow([epoch+1, avg_mse, avg_div])
            csv_file.flush()
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(model.module.state_dict(), f'mhd_diffusion_unet_ep{epoch+1}.pth')

    if global_rank == 0:
        csv_file.close()
        print("Distributed Training Complete!")

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
