#!/bin/bash


#SBATCH --output=tf_dmrg_output.txt
#SBATCH --error=tf_dmrg_error.txt
#SBATCH --mem=10g
#SBATCH --cpus-per-task=8

BOND_LEN=$1
NB_EPOCHS=$2
BATCH_SIZE=$3
LEARNING_RATE=$4

srun python3 dmrg_expt.py $BOND_LEN $NB_EPOCHS $BATCH_SIZE $LEARNING_RATE

