#!/bin/bash
#PBS -A DD_plasma_physics
#PBS -q preemptable
#PBS -l select=4:system=polaris
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:eagle
#PBS -N mhd_ddp_train
#PBS -o mhd_ddp.out
#PBS -e mhd_ddp.err

cd /eagle/DD_plasma_physics/CSE674_MHD_project
module use /soft/modulefiles
module load conda
conda activate base

# Determine the master node IP for MPI coordination
export MASTER_ADDR=$(cat $PBS_NODEFILE | head -1)
export MASTER_PORT=29500
export NNODES=$(wc -l < $PBS_NODEFILE)
export NGPU_PER_NODE=4
export WORLD_SIZE=$((NNODES * NGPU_PER_NODE))

echo "Master Node: $MASTER_ADDR"
echo "Total Nodes: $NNODES | Total GPUs: $WORLD_SIZE"

# Launch PyTorch DDP using mpiexec across the 4 nodes
mpiexec -n $WORLD_SIZE --ppn $NGPU_PER_NODE --depth=8 --cpu-bind depth \
    --env MASTER_ADDR=$MASTER_ADDR --env MASTER_PORT=$MASTER_PORT \
    --env WORLD_SIZE=$WORLD_SIZE \
    ./launch_pytorch.sh python train_ddp.py