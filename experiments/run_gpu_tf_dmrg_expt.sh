#!/bin/bash

#SBATCH --output=$HOME/tfgpu_dmrg_output.log
#SBATCH --error=$HOME/tfgpu_dmrg_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=10g

srun python3 dmrg_expt.py
