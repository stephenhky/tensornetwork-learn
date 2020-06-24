#!/bin/bash


#SBATCH --output=$HOME/tf_dmrg_output_$SLURM_JOB_ID.log
#SBATCH --error=$HOME/tf_dmrg_error_$SLURM_JOB_ID.log
#SBATCH --mem=10g
#SBATCH --cpus-per-task=8

srun python3 dmrg_expt.py

