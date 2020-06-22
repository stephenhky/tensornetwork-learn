#!/bin/bash

#SBATCH --output=tf_dmrg_output.txt
#SBATCH --error=tf_dmrg_error.txt
#SBATCH --mem=10g
#SBATCH --cpus-per-task=8

srun python3 dmrg_expt.py

