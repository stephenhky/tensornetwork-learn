#!/bin/bash

#SBATCH --output=$HOME/tf_dmrg_output_%A.log
#SBATCH --error=$HOME/tf_dmrg_error_%A.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=10g

srun python3 dmrg_expt.py
