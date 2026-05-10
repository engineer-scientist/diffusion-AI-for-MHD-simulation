#!/bin/bash
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --job-name=mhd_ddp_100ep
#SBATCH --output=scale_train_%j.out
#SBATCH --error=scale_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ss694@buffalo.edu

module load gcc/11.2.0
module load python/3.10.4 # Adjust based on CCR's current modules
# source your conda environment here if you build one:
# source activate my_mhd_env

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

echo "Starting distributed training on $SLURM_NNODES nodes ($WORLD_SIZE GPUs)..."
echo "Master Node: $MASTER_ADDR"

# Launch across all nodes using srun
srun --mpi=pmi2 \
     python -u train_ddp.py