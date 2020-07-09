#!/bin/bash

#SBATCH --output=torch_dmrg_output.txt
#SBATCH --error=torch_dmrg_error.txt
#SBATCH --mem=5g
#SBATCH --cpus-per-task=4

srun python3 dmrg_torchmps_expt.py

