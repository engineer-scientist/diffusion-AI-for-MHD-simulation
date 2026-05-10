#!/bin/bash
#PBS -A gpu_hack
#PBS -q preemptable
#PBS -l select=4:system=polaris
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -N mhd_ddp_scale
#PBS -o scale_train.out
#PBS -e scale_train.err

cd /eagle/DD_plasma_physics/CSE674_MHD_project

module use /soft/modulefiles
module load conda
conda activate base

# Polaris-specific networking variables to prevent MPI hanging:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_RX_MATCH_MODE=software

export NNODES=`wc -l < $PBS_NODEFILE`
export NRANKS_PER_NODE=4
export NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

# Identify the master node:
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500

echo "Starting distributed training on $NNODES nodes ($NTOTRANKS GPUs)..."

# Launch across all nodes:
mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE --depth=8 --cpu-bind depth \
    --env MASTER_ADDR=$MASTER_ADDR --env MASTER_PORT=$MASTER_PORT \
    --env WORLD_SIZE=$NTOTRANKS \
    ./launch_pytorch.sh python -u train_ddp.py