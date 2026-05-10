#!/bin/bash
#PBS -A DD_plasma_physics
#PBS -q preemptable
#PBS -l select=2:system=polaris
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:eagle
#PBS -N mhd_ddp_train
#PBS -o mhd_ddp.out
#PBS -e mhd_ddp.err

cd /eagle/DD_plasma_physics/CSE674_MHD_project
module use /soft/modulefiles
module load conda
conda activate base

# Extract the master node IP for torchrun
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500
export NNODES=$(wc -l < $PBS_NODEFILE)
export NGPUS_PER_NODE=4
export TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))

echo "Starting distributed training on $NNODES nodes ($TOTAL_GPUS GPUs)"

# Launch using PyTorch's native distributed runner via MPI
mpiexec -n $NNODES --ppn 1 --hostfile $PBS_NODEFILE \
    torchrun --nnodes=$NNODES \
             --nproc_per_node=$NGPUS_PER_NODE \
             --rdzv_id=674_mhd \
             --rdzv_backend=c10d \
             --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
             train.py