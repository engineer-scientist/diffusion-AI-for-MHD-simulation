# train_ddp.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
# import time

from MHD_dataset import MHDDataset
from unet_3D import UNet3D

def compute_divergence_loss(pred_x0):
    B = pred_x0[:, 4:7, :, :, :]
    dBx_dx = B[:, 0, :, :, 2:] - B[:, 0, :, :, :-2]
    dBy_dy = B[:, 1, :, 2:, :] - B[:, 1, :, :-2, :]
    dBz_dz = B[:, 2, 2:, :, :] - B[:, 2, :-2, :, :]
    div_B = (dBx_dx[:, 1:-1, 1:-1, :] + dBy_dy[:, 1:-1, :, 1:-1] + dBz_dz[:, :, 1:-1, 1:-1]) / 2.0
    return torch.mean(div_B ** 2)

def train(n_epochs):
    # --- DDP Initialization ---
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    if global_rank == 0:
        print(f"\n Starting Multi-Node DDP Training across {dist.get_world_size()} GPUs!")

    # --- Dataset & Distributed Sampler ---
    dataset = MHDDataset('data/train/')
    # The sampler ensures each GPU gets a different slice of the data
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler) 

    # --- Model & Optimizer ---
    model = UNet3D().to(device)
    # Wrap the model for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- Diffusion Schedule ---
    n_timesteps = 1000
    beta = torch.linspace(1e-4, 0.02, n_timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)


    # if global_rank == 0:
        # t1 = time.time()

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch) # Shuffles data differently each epoch
        model.train()
        epoch_mse, epoch_div = 0, 0
        
        for batch_x, batch_cond in dataloader:
            batch_x = batch_x.to(device)
            batch_cond = batch_cond.to(device)
            
            noise = torch.randn_like(batch_x).to(device)
            t = torch.randint(0, n_timesteps, (batch_x.shape[0],)).to(device)
            
            a_bar_t = alpha_bar[t].view(-1, 1, 1, 1, 1)
            noisy_batch = torch.sqrt(a_bar_t) * batch_x + torch.sqrt(1 - a_bar_t) * noise
            
            optimizer.zero_grad()
            # Pass both the noisy data AND the physical conditions (Ma, Ms)
            predicted_noise = model(noisy_batch, t, batch_cond)
            
            mse_loss = F.mse_loss(predicted_noise, noise)
            pred_x0 = (noisy_batch - torch.sqrt(1 - a_bar_t) * predicted_noise) / torch.sqrt(a_bar_t)
            div_loss = compute_divergence_loss(pred_x0)
            
            total_loss = mse_loss + (0.1 * div_loss)
            total_loss.backward()
            optimizer.step()
            
            epoch_mse += mse_loss.item()
            epoch_div += div_loss.item()

        # Print only from the master GPU to avoid duplicate logs
        if global_rank == 0:
            avg_mse = epoch_mse / len(dataloader)
            avg_div = epoch_div / len(dataloader)
            print(f"Epoch {epoch+1} / {n_epochs} | Avg MSE: {avg_mse:.4f} | Avg Div(B): {avg_div:.4f}")
            
            # Log to CSV for your plots: 
            with open('training_log_' + str(n_epochs) + '_epochs.csv', 'a') as f:
                f.write(f"{epoch+1}, {avg_mse}, {avg_div}\n")

            # Saving model weights once every 10 epochs: 
            # if ((epoch + 1) % 10) == 0:
                # torch.save(model.module.state_dict(), 'MHD_diffusion_scaled_' + str(n_epochs) + '_epochs.pth')
                # print("----- Model saved. -----")
    
    # All training epochs completed: 
    
    if global_rank == 0:
        # t2 = time.time()
        torch.save(model.module.state_dict(), 'MHD_diffusion_scaled_' + str(n_epochs) + '_epochs.pth')
        print("\n Distributed training complete! Model saved.")
        # print("Training time:", t2 - t1, "seconds. \n")

    dist.destroy_process_group()

if __name__ == "__main__":
    n_epochs = 30
    if int(os.environ.get('RANK', 0)) == 0 and not os.path.exists('training_log_' + str(n_epochs) + '_epochs.csv'):
        with open('training_log_' + str(n_epochs) + '_epochs.csv', 'w') as f:
            f.write("Epoch, MSE, DivB\n")
    train(n_epochs)
