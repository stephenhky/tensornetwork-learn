#!/bin/bash


#SBATCH --output=tn_fruit_output.txt
#SBATCH --error=tn_fruit_error.txt
#SBATCH --mem=5g
#SBATCH --cpus-per-task=8

LEARNINGRATE=$1
NBEPOCHS=$2
BATCHSIZE=$3
OUTPUTFILE=$4

python3 fruits_expt.py fruits/fruits.json --featurefilepath fruit_features.json --learning_rate $LEARNINGRATE --nbepochs $NBEPOCHS --batch_size $BATCHSIZE --output_file $OUTPUTFILE
