#!/bin/bash


#SBATCH --output=/lscratch/$SLURM_JOB_ID
#SBATCH --error=tf_dmrg_error.txt
#SBATCH --mem=10g
#SBATCH --cpus-per-task=8

BOND_LEN=$1
NB_EPOCHS=$2
BATCH_SIZE=$3
LEARNING_RATE=$4
OUTPUT_FILE=$5

srun python3 dmrg_expt.py $BOND_LEN $NB_EPOCHS $BATCH_SIZE $LEARNING_RATE > $OUTPUT_FILE

