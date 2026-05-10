Project: Physics-Informed Diffusion AI Models for MHD Simulations (Application in Nuclear Fusion Energy),
Course: CSE 674 - Advanced Machine Learning,
Semester: Spring 2026, 
Student: Sarthak Sharma (ss694@buffalo.edu), 
Institution: State University of New York at Buffalo.

Project Overview:
This repository contains the source code for a 3-dimensional, physics-informed score-based diffusion 
model designed to generate structurally accurate Magnetohydrodynamics (MHD) plasma simulations. 
The model utilizes a multi-modal 3D U-Net backbone and incorporates a Discrete Exterior Calculus (DEC) 
inspired loss penalty to enforce the divergence-free condition (∇·B = 0) of the generated magnetic fields. 

Included files in the ZIP package:
* train_ddp.py: The main training loop utilizing PyTorch Distributed Data Parallel (DDP). It handles multi-node scaling, noise scheduling, and the computation of the divergence-free physics loss penalty.
* unet_3D.py: The architecture definition of the conditional 3D U-Net. It processes 7-channel 3D tensors (Density, Velocity, Magnetic Field) and accepts Mach number conditions via MLP embeddings.
* MHD_dataset.py: A custom PyTorch `Dataset` class designed to dynamically scan and load multi-trajectory HDF5 files into memory.
* generate_plasma.py: The inference script. It performs the 1,000-step reverse diffusion process from pure Gaussian noise to a structured plasma state based on user-defined Mach parameters, and outputs a 2D visualization slice.
* submit_ddp_scale.sh: The PBS Pro batch submission script used to scale the training across 4 compute nodes (8x NVIDIA A100 GPUs) on the ALCF Polaris supercomputer.
* MHD_diffusion_scaled_30_epochs.pth: Saved model weights. 
* README.txt (this file).

Dependencies:
The code requires Python 3.10+ and the following libraries:
* torch (PyTorch with CUDA support recommended), 
* h5py (For reading the massive HDF5 datasets), 
* numpy, 
* matplotlib (For generating inference visualizations), 
* pandas (For tracking training logs), 
* mpi4py (Required for PyTorch DDP via NCCL backend).

---------------------

How to Run the Code:

1. Training the Model (Distributed Data Parallel):
To train the model from scratch on a supercomputing cluster (like ALCF Polaris), ensure your environment 
modules are loaded and submit the PBS script:
qsub submit_ddp_scale.sh

Note: The script expects the dataset to be located in a data/train/ directory relative to the execution path.

2. Running Inference (Generating Plasma States):
To generate a novel plasma state using the trained weights, run the generate_plasma.py script. You can 
condition the AI generation by passing specific Mach (--ma) and Sonic Mach (--ms) parameters.

Example (High Turbulence):
python generate_plasma.py --ma 2.0 --ms 7.0

Example (Low Turbulence):
python generate_plasma.py --ma 0.7 --ms 0.5

The script will output a .png file containing the heat map of the plasma density and the continuous 
streamlines of the magnetic field.

(Note: The mhd_diffusion_scaled_30_epochs.pth weights file is required to run inference.).

---------------------

Acknowledgments and Data Source:

Dataset: 
The training data utilized in this project is from the MHD_64 subset of "The Well", a large-scale 
dataset for scientific machine learning.
Polymathic AI Collaborative. (2023). The Well: A Large-Scale Dataset for Scientific Machine Learning in 
Fluid Dynamics and Magnetohydrodynamics. Available at: https://polymathic-ai.org/the_well/

Computation Resources: 
Computational resources were provided by the Director's Discretionary Allocation at the Argonne 
Leadership Computing Facility (ALCF), Argonne National Laboratory.
