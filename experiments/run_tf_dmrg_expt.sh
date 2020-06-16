#!/bin/bash

#SBATCH --output=$HOME/tf_dmrg_output.log
#SBATCH --error=$HOME/tf_dmrg_error.log
#SBATCH --mem=10g

srun python3 dmrg_expt.py
