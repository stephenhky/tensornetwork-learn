#!/bin/bash

#SBATCH --output=tfgpu_dmrg_output.txt
#SBATCH --error=tfgpu_dmrg_error.txt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=lscratch:10,gpu:k80:1
#SBATCH --mem=10g

srun python3 dmrg_expt.py
